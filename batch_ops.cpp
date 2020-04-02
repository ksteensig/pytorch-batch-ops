
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

std::tuple<torch::Tensor, torch::Tensor> batch_symeig_forward(
    torch::Tensor X, bool is_sort, double tol = 1e-7, int max_sweeps = 100) {
  auto handle = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

  auto U = X.clone();  // U will contain the eigenvectors

  auto batch_size = X.size(0);
  auto L = X.size(1);

  // D contains the eigenvalus
  auto D = torch::zeros({batch_size, L}).to(torch::kCUDA);

  int lwork = 0;

  auto params =
      unique_allocate(cusolverDnCreateSyevjInfo, cusolverDnDestroySyevjInfo);
  auto status = cusolverDnXsyevjSetTolerance(params.get(), tol);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXsyevjSetMaxSweeps(params.get(), max_sweeps);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXsyevjSetSortEig(params.get(), is_sort);
  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

  auto status2 = cusolverDnSsyevjBatched_bufferSize(
      handle.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, L,
      U.data<float>(), L, D.data<float>(), &lwork, params.get(), batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status2);

  auto d_work = unique_cuda_ptr<float>(lwork);
  auto d_info = unique_cuda_ptr<int>(batch_size);

  status2 = cusolverDnSsyevjBatched(handle.get(), CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER, L, U.data<float>(),
                                    L, D.data<float>(), d_work.get(), lwork,
                                    d_info.get(), params.get(), batch_size);

  AT_CHECK(CUSOLVER_STATUS_SUCCESS == status2);

  // CUDA works with column major, so the matrices have to be made row major
  // before using them in PyTorch
  return std::make_pair(D, U);
}

// this backward is based on the backward from symeig_backward Functions.cpp
// https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/Functions.cpp
torch::Tensor batch_symeig_backward(const std::vector<torch::Tensor>& grads,
                                    const torch::Tensor& self,
                                    const torch::Tensor& D,
                                    const torch::Tensor& U) {
  auto dD = grads[0];
  auto dU = grads[1];

  auto Ut = U.transpose(1, 2);

  torch::Tensor result;

  // diagonal of all eigenvalue matrices
  auto diag = D.diagonal(0, 1, 2);

  if (dU.defined()) {
    auto F = diag.unsqueeze(-2) - diag.unsqueeze(-1);
    F.diagonal(0, -2, -1).fill_(INFINITY);
    F.pow_(-1);
    F.mul_(at::matmul(Ut, dU));
    result = at::matmul(U, at::matmul(F, Ut));
  }
  /*
      else {
        result = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }
  */
  if (dD.defined()) {
    result.add_(at::matmul(
        at::matmul(U,
                   at::diag_embed(dU, /*offset=*/0, /*dim1=*/-2, /*dim2=*/-1)),
        Ut));
  }

  return result.add(result.transpose(-2, -1)).mul_(0.5);
}

// https://j-towns.github.io/papers/svd-derivative.pdf
//
// This makes no assumption on the signs of sigma.
torch::Tensor batch_csvd_backward(const std::vector<at::Tensor>& grads,
                                  const at::Tensor& self, bool some,
                                  bool compute_uv, const at::Tensor& raw_u,
                                  const at::Tensor& sigma,
                                  const at::Tensor& raw_v) {
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

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
  } else {
    sigma_term = at::zeros({1}, self.options()).expand_as(self);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(1, 2);
  auto im = at::eye(m, self.options());  // work if broadcast
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed();
  auto sigma_mat_inv = sigma.pow(-1).diag_embed();
  auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
  auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(0, -2, -1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term =
        u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
    if (m > k) {
      u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
    }
    u_term = u_term.bmm(vt);
  } else {
    u_term = at::zeros({1}, self.options()).expand_as(self);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(1, 2);
    v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
    if (n > k) {
      v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
    }
    v_term = u.bmm(v_term);
  } else {
    v_term = at::zeros({1}, self.options()).expand_as(self);
  }

  return u_term + sigma_term + v_term;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig_forward", &batch_symeig_forward,
        "cusolver based batch symeig implementation");

  m.def("batch_csvd_backward", &batch_csvd_backward,
        "autograd support for the batch csvd operation");
}
