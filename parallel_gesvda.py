import torch
import torch_batch_ops_cpp
import math

def csvd(X):
    size = list(X.size())
    m = size[1]
    n = size[2]
    batch_size = size[0]

    p = 20
    k = math.floor(n*0.10)
    l = k+p # estimate a low rank approx that is 10% of  with p oversampling

    #Phi = torch.randint(0,2,(batch_size, l, m),device='cuda:0',dtype=torch.float32)
    #Y = Phi.matmul(X)
    Y = X[:,:l,:]
    Yt = Y.transpose(1,2)

    B = Y.matmul(Yt)
    B = B.add(B.transpose(1,2))
    B.mul_(0.5)

    index = torch.range(l-1, 0, -1, dtype=torch.long).to('cuda:0', non_blocking=True)
    
    D,T = torch_batch_ops_cpp.batch_symeig_cpp(B, True, 1e-7, 20)
    D = D.index_select(dim=1, index=index)
    T = T.index_select(dim=1, index=index).transpose(1,2)
    S_ = D[:,:k].pow(-0.5).diag_embed(0, 1, 2)

    V_ = Yt.matmul(T[:,:,:k]).matmul(S_)
    U_ = X.matmul(V_)

    U,S,Q = torch_batch_ops_cpp.batch_gesvda_cpp(U_)

    V = V_.matmul(Q)

    return U, S, V

import time

n = 5

def mat_copy():
    X = torch.randn(20, 10000, 784).to('cuda:0')

def csvd_fun():
    X = torch.randn(20, 10000, 784).to('cuda:0')
    U,S,V = csvd(X)

def svd_fun():
    X = torch.randn(20, 10000, 784).to('cuda:0')
    for x in X:
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
