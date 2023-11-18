import torch 
import numpy as np
import time


x = torch.rand([20000,3000])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x_cuda = x.to(device)

start = time.time()
y = (x - x + x * 10.0) ** 2
print(1000*(time.time() - start))

start = time.time()
y_cuda = (x_cuda - x_cuda + x_cuda * 10.0) ** 2
print(1000*(time.time() - start))
