import math
import torch

def csvd(X):
    size = list(X.size())
    m = size[0]
    n = size[1]

    p = 20
    k = math.floor(n*0.10)
    l = k+p # estimate a low rank approx. that is 10% singular values with 20 oversampling

    Y = X[:l,:]
    Yt = Y.t()

    B = Y.matmul(Yt)
    B = B.add(B.t())
    B.mul_(0.5)

    index = torch.range(l-1, 0, -1, dtype=torch.long, device='cuda:0')
        
    D,T = B.symeig(eigenvectors=True)
    D = D.index_select(0,index=index)
    T = T.index_select(0,index=index).t()
    S_ = D[:k].pow(-0.5).diag()
    
    V_ = Yt.matmul(T[:,:k]).matmul(S_)
    U_ = X.matmul(V_)
    
    U,S,Q = U_.svd(compute_uv=True)

    V = V_.matmul(Q)

    return U, S, V

import time

N = 10
B = 25

X = torch.randn(10000, 784).to('cuda:0')
torch.cuda.synchronize()
U,S,V = csvd(X)
torch.cuda.synchronize()

for _ in range(N):
    X = torch.randn(B, 10000, 784).to('cuda:0')
    torch.cuda.synchronize()

    csvd_start = time.time()
    for x in X:
        U,S,V = csvd(x)
    torch.cuda.synchronize()
    csvd_end = time.time()
    csvd_time = (csvd_end-csvd_start)
    print(csvd_time)
