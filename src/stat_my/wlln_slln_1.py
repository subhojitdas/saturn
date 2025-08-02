import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


rng = np.random.default_rng(0)

mu = 0.5
epsilons = [ 0.1, 0.05 ]
sample_sizes = [10, 50, 100, 500, 1000, 3000, 10000]
trials = 2000

records = []

for n in sample_sizes:
    X = rng.binomial(1, 0.5, size=(trials, n))
    means = X.mean(axis=1)
    for eps in epsilons:
        prob_far = np.mean(np.abs(means - mu) > eps)
        records.append({"n": n, "epsilon": eps, "estimated P(|mean-mu| > eps)": prob_far})

df = pd.DataFrame(records)

print("Convergence in probability estimates", df)

for eps in epsilons:
    subset = df[df["epsilon"] == eps].sort_values("n")
    plt.figure()
    plt.plot(subset["n"], subset["estimated P(|mean-mu| > eps)"], marker='o')
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.title(f"Estimated P(|sample mean − mu| > {eps}) vs n (Bernoulli(0.5))")
    plt.xlabel("n (log scale)")
    plt.ylabel("estimated probability")
    plt.show()

