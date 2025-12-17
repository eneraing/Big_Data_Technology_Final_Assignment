import numpy as np
import random
import matplotlib.pyplot as plt

def generate_data(length=500):
    true, obs = [], []
    for _ in range(length):
        x = np.random.normal(50, 10)
        true.append(x)
        obs.append(x * 6 if random.random() < 0.1 else x)
    return np.array(true), np.array(obs)

def adaptive(obs, w=20):
    out, repairs = [], 0
    for i, x in enumerate(obs):
        if i < w:
            out.append(x)
            continue
        mu = np.mean(out[i-w:i])
        std = np.std(out[i-w:i]) + 1e-6
        z = abs(x - mu) / std
        if z > 3:
            out.append(mu)
            repairs += 1
        else:
            out.append(x)
    return np.array(out), repairs

true, obs = generate_data()
cleaned, repair_cnt = adaptive(obs)

lambdas = [0, 0.5, 1, 2, 4]
costs = []

for lam in lambdas:
    cost = lam * repair_cnt + abs(np.mean(cleaned) - np.mean(true))
    costs.append(cost)

plt.figure()
plt.plot(lambdas, costs, marker='o')
plt.xlabel("Repair Cost Weight λ")
plt.ylabel("Total Cost")
plt.title("Cleaning Cost vs Data Quality Trade-off")
plt.show()
