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

n = 5

def mat_copy():
    x = torch.randn(20, 10000, 784).to('cuda:0')

def csvd_fun():
    X = torch.randn(20, 10000, 784).to('cuda:0')
    for x in X:
        U,S,V = csvd(x)

def svd_fun():
    x = torch.randn(20, 10000, 784).to('cuda:0')
    torch.cuda.synchronize()
    for x in x:
        U,S,V = x.svd()

torch.cuda.synchronize()
copy_start = time.time()
for _ in range(n):
    mat_copy()
torch.cuda.synchronize()
copy_end = time.time()

copy_time = (copy_end-copy_start)/n

torch.cuda.synchronize()
csvd_start = time.time()
for _ in range(n):
    csvd_fun()
torch.cuda.synchronize()
csvd_end = time.time()

csvd_time = (csvd_end-csvd_start)/n - copy_time


torch.cuda.synchronize()
svd_start = time.time()
for _ in range(n):
    svd_fun()
torch.cuda.synchronize()
svd_end = time.time()

svd_time = (svd_end-svd_start)/n - copy_time

print(copy_time)
print(csvd_time)
print(svd_time)
print(svd_time/csvd_time)
