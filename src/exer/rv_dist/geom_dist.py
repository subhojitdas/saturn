import matplotlib.pyplot as plt
import numpy as np

p = 0.6
q = 1 - p

k = np.arange(1, 11)
pmf_values = (q ** (k - 1)) * p

plt.figure(figsize=(7, 5))
plt.stem(k, pmf_values, basefmt=" ")
plt.ylabel("P(X = k)")
plt.xticks(k)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for k, prob in zip(k, pmf_values):
    plt.text(k, prob + 0.01, f"{prob:.4f}", ha='center', fontsize=9)

plt.show()