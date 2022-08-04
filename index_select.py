import time

import numpy as np
import torch

INDEX = 100000
NELE = 1000
a = torch.rand(INDEX, NELE)
index = np.random.randint(INDEX-1, size=INDEX*8)
b = torch.from_numpy(index)

res = a.index_select(0, b)
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    res = a.index_select(0, b)
torch.cuda.synchronize()
print('{}s'.format(time.time() - start))