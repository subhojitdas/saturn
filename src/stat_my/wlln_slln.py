import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n_max = 10000
coin_flip = rng.binomial(1, 0.5, size=n_max)
running_mean = np.cumsum(coin_flip) / np.arange(1, n_max + 1)


plt.figure()
plt.plot(running_mean, linewidth=1)
plt.axhline(0.5, linestyle='--')
plt.title("Running sample mean of a fair coin (0/1)")
plt.xlabel("n (number of flips)")
plt.ylabel("sample mean up to n")
plt.show()