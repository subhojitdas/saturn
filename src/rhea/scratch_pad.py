import numpy as np

# random_array = np.random.rand(2, 3)
# print(random_array)
# random_array = np.c_[np.ones((2, 1)) * 1.1123, random_array]
# print(random_array)
#
# X = np.random.normal(0, 1, (100, 2))
# print(np.get_include())

from src.titan.low.decision_tree import best_split
arr = np.random.rand(18, 6)

best_split(arr)

