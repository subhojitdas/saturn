import random
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

res = f(3.0)
print(res)

xs = np.arange(-5.0, 5.0, 0.25)
ys = f(xs)
plt.plot(xs, ys)
plt.show()

h = 0.00001
x = 2/3
(f(x+h) - f(x))/h
