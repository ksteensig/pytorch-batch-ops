import math
import torch

def csvd(X):
    size = list(X.size())
    m = size[1]
    n = size[2]
    batch_size = size[0]

    p = 20
    k = math.floor(n*0.10)
    l = k+p # estimate a low rank approx. that is 10% singular values with 20 oversampling

    Y = X[:,:l,:]
    Yt = Y.transpose(1,2)

    B = Y.matmul(Yt)
    B = B.add(B.transpose(1,2))
    B.mul_(0.5)

    index = torch.range(l-1, 0, -1, dtype=torch.long, device='cuda:0')

    D = torch.empty(batch_size, l, device=torch.device("cuda:0"))
    T = torch.empty(batch_size, l, l, device=torch.device("cuda:0"))
    
    for i in range(batch_size):
        D[i],T[i] = B[i].symeig(eigenvectors=True)
   
    D = D.index_select(1,index=index)
    T = T.index_select(1,index=index).transpose(1,2)
    S_ = D[:,:k].pow(-0.5).diag_embed(0,1,2)

    V_ = Yt.matmul(T[:,:,:k]).matmul(S_)
    U_ = X.matmul(V_)

    U = torch.empty(batch_size, m, k, device=torch.device("cuda:0"))
    S = torch.empty(batch_size, k, device=torch.device("cuda:0"))
    Q = torch.empty(batch_size, k, k, device=torch.device("cuda:0"))

    for i in range(batch_size):
        U[i],S[i],Q[i] = U_[i].svd(compute_uv=True) 
    
    V = V_.matmul(Q)

    return U, S, V

import time

N = 10
B = 25

X = torch.randn(B, 10000, 784).to('cuda:0')
torch.cuda.synchronize()
U,S,V = csvd(X)
torch.cuda.synchronize()

for _ in range(N):
    X = torch.randn(B, 10000, 784).to('cuda:0')
    torch.cuda.synchronize()

    csvd_start = time.time()
    U,S,V = csvd(X)
    torch.cuda.synchronize()
    csvd_end = time.time()
    csvd_time = (csvd_end-csvd_start)
    print(csvd_time)
