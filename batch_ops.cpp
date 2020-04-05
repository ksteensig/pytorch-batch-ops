#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

template <int success = CUSOLVER_STATUS_SUCCESS, class T, class Status>
std::unique_ptr<T, Status (*)(T*)> unique_allocate(Status(allocator)(T**),
                                                   Status(deleter)(T*)) {
  T* ptr;
  auto stat = allocator(&ptr);
  AT_CHECK(stat == success);
  return {ptr, deleter};
}

template <class T>
std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
  T* ptr;
  auto stat = cudaMalloc(&ptr, sizeof(T) * len);
  AT_CHECK(stat == cudaSuccess);
  return {ptr, cudaFree};
}

std::tuple<torch::Tensor, torch::Tensor> batch_symeig(torch::Tensor X,
                                                      bool is_sort,
                                                      double tol = 1e-7,
                                                      int max_sweeps = 100) {
  auto handle = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

  auto batch_size = X.size(0);
  auto L = X.size(1);

  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat64)
    .device(torch::kCUDA, 0);

  auto U = X;

  // D contains the eigenvalus
  auto D = torch::empty({batch_size, L}, options);

  int lwork = 0;

  auto params =
      unique_allocate(cusolverDnCreateSyevjInfo, cusolverDnDestroySyevjInfo);
  auto status = cusolverDnXsyevjSetTolerance(params.get(), tol);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXsyevjSetMaxSweeps(params.get(), max_sweeps);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXsyevjSetSortEig(params.get(), is_sort);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

  auto status2 = cusolverDnDsyevjBatched_bufferSize(
      handle.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, L,
      U.data<double>(), L, D.data<double>(), &lwork, params.get(), batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status2);

  auto d_work = unique_cuda_ptr<double>(lwork);
  auto d_info = unique_cuda_ptr<int>(batch_size);

  status2 = cusolverDnDsyevjBatched(handle.get(), CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER, L, U.data<double>(),
                                    L, D.data<double>(), d_work.get(), lwork,
                                    d_info.get(), params.get(), batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status2);

  // CUDA works with column major, so the matrices have to be made row major
  // before using them in PyTorch
  return std::make_pair(D, U);
}

// solve U S V = svd(A)  a.k.a. syevj, where A (b, m, n), U (b, m, m), S (b,
// min(m, n)), V (b, n, n) see also
// https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batch_gesvda(
    torch::Tensor A) {
  auto handle = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

  auto X = A.contiguous().transpose(1,2).contiguous().transpose(1,2);

  auto batch_size = X.size(0);
  auto height = X.size(1);
  auto width = X.size(2);

  auto rank = width;

  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat64)
    .device(torch::kCUDA, 0);

  // CUDA uses column major, so U and V are created as column major matrices
  // they are then transposed for pytorch, as transposing does not move any memory
  auto U = torch::empty({batch_size, rank, height}, options).transpose(1,2);
  auto S = torch::empty({batch_size, rank}, options);
  auto V = torch::empty({batch_size, width, rank}, options).transpose(1,2);

  auto ldx = 0;
  auto ldu = 0;
  auto ldv = 0;

  auto strideX = 0;
  auto strideS = 0;
  auto strideU = 0;
  auto strideV = 0;

  if (batch_size > 1) {
    ldx = height;
    ldu = height;
    ldv = width; 
    strideX = X[1].storage_offset();
    strideS = S[1].storage_offset();
    strideU = U[1].storage_offset();
    strideV = V[1].storage_offset();
  }

  int lwork = 0;

  auto status = cusolverDnDgesvdaStridedBatched_bufferSize(
      handle.get(), CUSOLVER_EIG_MODE_VECTOR, rank, height, width, X.data<double>(),
      ldx, strideX, S.data<double>(), strideS, U.data<double>(), ldu, strideU,
      V.data<double>(), ldv, strideV, &lwork, batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

  auto d_work = unique_cuda_ptr<double>(lwork);
  auto d_info = unique_cuda_ptr<int>(batch_size);

  status = cusolverDnDgesvdaStridedBatched(
      handle.get(), CUSOLVER_EIG_MODE_VECTOR, rank, height, width, X.data<double>(),
      ldx, strideX, S.data<double>(), strideS, U.data<double>(), ldu, strideU,
      V.data<double>(), ldv, strideV, d_work.get(), lwork, d_info.get(), NULL,
      batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

  return std::make_tuple(U, S, V);
}

// https://j-towns.github.io/papers/svd-derivative.pdf
// This makes no assumption on the signs of sigma.
torch::Tensor batch_csvd_backward(const std::vector<torch::Tensor>& grads,
                                  const torch::Tensor& self, bool some,
                                  bool compute_uv, const torch::Tensor& raw_u,
                                  const torch::Tensor& sigma,
                                  const torch::Tensor& raw_v) {
  AT_CHECK(compute_uv,
           "csvd_backward: Setting compute_uv to false in torch.svd doesn't "
           "compute singular matrices, ",
           "and hence we cannot compute backward. Please use "
           "torch.svd(compute_uv=True)");

  // A [b, m, n]
  // auto b = self.size(0);
  auto m = self.size(1);
  auto n = self.size(2);
  auto k = sigma.size(1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(2, 0, k);
    v = raw_v.narrow(2, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(2, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(2, 0, k);
    }
  }
  auto vt = v.transpose(1, 2);

  torch::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
  } else {
    sigma_term = torch::zeros({1}, self.options()).expand_as(self);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(1, 2);
  auto im = torch::eye(m, self.options());  // work if broadcast
  auto in = torch::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed();
  auto sigma_mat_inv = sigma.pow(-1).diag_embed();
  auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
  auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(0, -2, -1).fill_(INFINITY);
  F = F.pow(-1);

  torch::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term =
        u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
    if (m > k) {
      u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
    }
    u_term = u_term.bmm(vt);
  } else {
    u_term = torch::zeros({1}, self.options()).expand_as(self);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(1, 2);
    v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
    if (n > k) {
      v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
    }
    v_term = u.bmm(v_term);
  } else {
    v_term = torch::zeros({1}, self.options()).expand_as(self);
  }

  return u_term + sigma_term + v_term;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig_cpp", &batch_symeig,
        "cusolver based batch symeig implementation");
  m.def("batch_gesvda_cpp", &batch_gesvda,
        "batch SVD using the cuSOLVER gesvda");
  m.def("batch_csvd_backward", &batch_csvd_backward,
        "autograd support for the batch csvd operation");
}
