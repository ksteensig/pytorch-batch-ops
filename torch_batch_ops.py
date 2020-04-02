import torch
import torch_batch_ops_cpp


class _BatchSymeigFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, X):
        D, U = torch_batch_ops_cpp.batch_symeig_forward(X, True, 1e-7, 20)
        
        self.save_for_backward(X, D, U)

        return D, U

    @staticmethod
    def backward(self, dD, dU):
        X, D, U = self.saved_variables

        grad_out = torch_batch_ops_cpp.batch_symeig_backward(
            [dD, dU], X, D, U
        )

        return grad_out


def batch_symeig(X):
    return _BatchSymeigFunction.apply(X)
