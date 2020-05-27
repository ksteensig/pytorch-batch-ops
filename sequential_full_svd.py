import torch
import time

N = 10
B = 25

X = torch.randn(10000, 784).to('cuda:0')
torch.cuda.synchronize()
U,S,V = X.svd()
torch.cuda.synchronize()

for _ in range(N):
    X = torch.randn(B, 10000, 784).to('cuda:0')
    torch.cuda.synchronize()

    csvd_start = time.time()
    for x in X:
        U,S,V = torch.svd(x,compute_uv=True)
    torch.cuda.synchronize()
    csvd_end = time.time()
    csvd_time = (csvd_end-csvd_start)
    print(csvd_time)
