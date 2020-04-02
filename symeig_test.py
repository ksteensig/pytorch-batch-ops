import torch as t
import torch_batch_ops as tbo

X = t.rand(2,50,50).to('cuda:0')
D,U = tbo.batch_symeig_forward(X, True, 10**-7, 20)

print(D.transpose(0,1))
print(U.transpose(0,1))


