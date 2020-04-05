import torch
import torch_batch_ops_cpp
import math

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

class BatchcSVDFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, X):
        size = list(X.size())
        m = size[1]
        n = size[2]
        batch_size = size[0]

        p = 10
        k = math.floor(n/10)
        l = k+p # estimate a low rank approx that is 10% of  with p oversampling

        Phi = torch.zeros(batch_size, l, m, device='cuda:0')
        Phi = torch.randint_like(Phi,0,2,device='cuda:0')
        #print(Phi)

        Y = Phi.matmul(X)
        Yt = Y.transpose(1,2)

        B = Y.matmul(Yt)
        B = B.add(B.transpose(1,2))
        B.mul_(0.5)

        D,T = torch_batch_ops_cpp.batch_symeig_cpp(B, True, 1e-7, 20)
        D,index = torch.sort(D, dim=1, descending=True)
        T = T.index_select(dim=1, index=index[0]).transpose(1,2)
        S_ = D[:,:k].pow(-0.5).diag_embed(0, 1, 2)

        V_ = Yt.matmul(T[:,:,:k]).matmul(S_)

        U_ = X.matmul(V_)

        U,S,Q = torch_batch_ops_cpp.batch_gesvda_cpp(U_) 

        V = V_.matmul(Q)

        return U, S, V

    @staticmethod
    def backward(self, grad_u, grad_s, grad_v):
        x, U, S, V = self.saved_variables

        grad_out = torch_batch_ops_cpp.batch_csvd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )
        return grad_out


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N]
    return:
        U, S, V = batch_svd(x) where x = USV^T
    """
    return BatchcSVDFunction.apply(x)


