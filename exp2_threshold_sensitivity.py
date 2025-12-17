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

def adaptive(obs, w=20, z_th=3):
    out = []
    for i, x in enumerate(obs):
        if i < w:
            out.append(x)
            continue
        mu = np.mean(out[i-w:i])
        std = np.std(out[i-w:i]) + 1e-6
        z = abs(x - mu) / std
        out.append(mu if z > z_th else x)
    return np.array(out)

true, obs = generate_data()
thresholds = [1.5, 2, 3, 4, 5]
shifts = []

for t in thresholds:
    cleaned = adaptive(obs, z_th=t)
    shifts.append(abs(np.mean(cleaned) - np.mean(true)))

plt.figure()
plt.plot(thresholds, shifts, marker='o')
plt.xlabel("Z-score Threshold")
plt.ylabel("Mean Shift")
plt.title("Sensitivity Analysis of Adaptive Threshold")
plt.show()
