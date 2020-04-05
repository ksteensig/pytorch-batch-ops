import torch
import torch_batch_ops as tbo
import torch_batch_ops_cpp as tboc
import pandas as pd
from timeit import timeit

torch.set_default_tensor_type(torch.DoubleTensor)

data = torch.from_numpy(pd.read_csv('mnist_1_61441.csv', header=None).to_numpy()).to('cuda:0')

X = torch.zeros(10, 10000, 784, dtype=torch.float64, device='cuda:0')

for i in range(10):
    X[i] = data.clone()

def fun_csvd():
    U,S,V = tbo.batch_svd(X)
    print(S)

def fun_svd():
    U,S,V = X[0].svd()

fun_csvd()

'''
if __name__=='__main__':
    from timeit import Timer
    tcsvd = Timer(lambda: fun_csvd())
    tsvd = Timer(lambda: fun_svd())
    print(tcsvd.timeit(number=10))
    print(tsvd.timeit(number=10))
'''
