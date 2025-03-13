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

shape = (3, 4, 2, 3)
rt = torch.rand(shape)
ot = torch.ones(shape)
zt = torch.zeros(shape)

print(f"Random Tensor: {rt}")
print(f"Ones Tensor: {ot}")
print(f"Zeros Tensor: {zt}")

t = torch.rand(3, 4)

print(f"Shape of tensor: {t.shape}")
print(f"Datatype of tensor: {t.dtype}")
print(f"Device tensor is stored on: {t.device}")

print(torch.cuda.is_available())

ot.add_(5)
print(ot)