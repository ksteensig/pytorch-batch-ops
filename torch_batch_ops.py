import torch
import torch_batch_ops_cpp
import math

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

        #Phi = torch.empty(batch_size, l, m, device='cuda:0')
        Phi = torch.randint(0,2,(batch_size, l, m),device='cuda:0',dtype=torch.float64)
        #print(Phi)

        Y = Phi.matmul(X)
        Yt = Y.transpose(1,2)

        B = Y.matmul(Yt)
        B = B.add(B.transpose(1,2))
        B.mul_(0.5)

        index = torch.range(l-1, 0, -1, dtype=torch.long, device='cuda:0')
        
        D,T = torch_batch_ops_cpp.batch_symeig_cpp(B, True, 1e-7, 20)
        D = D.index_select(dim=1, index=index)
        T = T.index_select(dim=1, index=index).transpose(1,2)
        S_ = D[:,:k].pow(-0.5).diag_embed(0, 1, 2)

        V_ = Yt.matmul(T[:,:,:k]).matmul(S_)
        U_ = X.matmul(V_)

        U,S,Q = torch_batch_ops_cpp.batch_gesvda_cpp(U_) 

        V = V_.matmul(Q)

        self.save_for_backward(X, U, S, V)

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


