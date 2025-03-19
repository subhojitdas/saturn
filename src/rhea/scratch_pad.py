import numpy as np

random_array = np.random.rand(2, 3)
print(random_array)
random_array = np.c_[np.ones((2, 1)) * 1.1123, random_array]
print(random_array)