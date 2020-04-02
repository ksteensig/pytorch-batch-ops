
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
  return std::make_pair(D.transpose(0, 1), U.transpose(0, 1));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig_forward", &batch_symeig_forward,
        "cusolver based batch symeig implementation");

  m.def("batch_symeig_backward", &batch_symeig_backward,
        "autograd support for the batch symeig operation");
}
