import torch
import numpy as np

data = [[8, 10], [2, 4]]

# x_data = torch.tensor(data)
# print(x_data)
# print(x_data.shape)
#
# np_array = np.array(data)
# print(np_array)
# x_np = torch.from_numpy(np_array)
# print(x_np)
#
#
# x_ones = torch.ones_like(x_data)
# print('x_ones', x_ones)
# x_rand = torch.rand_like(x_data, dtype=torch.float32)
# print('x_rand', x_rand)

# shape = (3, 4, 2, 3)
# rt = torch.rand(shape)
# ot = torch.ones(shape)
# zt = torch.zeros(shape)
#
# print(f"Random Tensor: {rt}")
# print(f"Ones Tensor: {ot}")
# print(f"Zeros Tensor: {zt}")
#
# t = torch.rand(3, 4)
#
# print(f"Shape of tensor: {t.shape}")
# print(f"Datatype of tensor: {t.dtype}")
# print(f"Device tensor is stored on: {t.device}")
#
# print(torch.cuda.is_available())
#
# ot.add_(5)
# print(ot)

# r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
# print('A random matrix, r:')
# print(r)
#
# # Common mathematical operations are supported:
# print('\nAbsolute value of r:')
# print(torch.abs(r))
#
# print('\nInverse sine of r:')
# print(torch.asin(r))
#
# print('\nDeterminant of r:')
# print(torch.det(r))
# print('\nSingular value decomposition of r:')
# print(torch.svd(r))
#
# print('\nAverage and standard deviation of r:')
# print(torch.std_mean(r))
# print('\nMaximum value of r:')
# print(torch.max(r))

# torch.manual_seed(1729)
# random1 = torch.rand(2, 3)
# print(random1)
#
# random2 = torch.rand(2, 3)
# print(random2)
#
# torch.manual_seed(1729)
# random3 = torch.rand(2, 3)
# print(random3)
#
# random4 = torch.rand(2, 3)
# print(random4)

# one = torch.ones(4, 2)
# print(one)
#
# b = torch.ones(1, 2) * 5
# print(b)
#
# c = one * b
# print(c)

# common functions
# a = torch.rand(2, 4) * 2 - 1
# print('Common functions:')
# print(torch.abs(a))
# print(torch.ceil(a))
# print(torch.floor(a))
# print(torch.clamp(a, -0.5, 0.5))

# vector and linear algebra operations
# v1 = torch.tensor([1., 0., 0.])         # x unit vector
# v2 = torch.tensor([0., 1., 0.])         # y unit vector
# m1 = torch.rand(2, 2)                   # random matrix
# m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix
#
# print('\nVectors & Matrices:')
# print(torch.linalg.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
# print(m1)
# m3 = torch.linalg.matmul(m1, m2)
# print(m3)                  # 3 times m1
# print(torch.linalg.svd(m3))
#

## Accelerator

# if torch.accelerator.is_available():
#     print('We have an accelerator!')
# else:
#     print('Sorry, CPU only.')
#
#
# if torch.accelerator.is_available():
#     gpu_rand = torch.rand(2, 2, device=torch.accelerator.current_accelerator())
#     print(gpu_rand)
# else:
#     print('Sorry, CPU only.')
#
# print(torch.accelerator.device_count())

my_device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
cpu_device = torch.device('cpu')
y = x.to(cpu_device)
print(y)

z = torch.rand(2, 2, device='cuda')
