import numpy as np
import random
import matplotlib.pyplot as plt

def generate_data(length=500, scale=5):
    true, obs = [], []
    for _ in range(length):
        x = np.random.normal(50, 10)
        true.append(x)
        obs.append(x * scale if random.random() < 0.1 else x)
    return np.array(true), np.array(obs)

def adaptive(obs, w=20):
    out = []
    for i, x in enumerate(obs):
        if i < w:
            out.append(x)
            continue
        mu = np.mean(out[i-w:i])
        std = np.std(out[i-w:i]) + 1e-6
        z = abs(x - mu) / std
        out.append(mu if z > 3 else x)
    return np.array(out)

scales = [3, 5, 8, 10]
shifts = []

for s in scales:
    true, obs = generate_data(scale=s)
    cleaned = adaptive(obs)
    shifts.append(abs(np.mean(cleaned) - np.mean(true)))

plt.figure()
plt.plot(scales, shifts, marker='o')
plt.xlabel("Anomaly Magnitude Scale")
plt.ylabel("Mean Shift")
plt.title("Effect of Noise Intensity")
plt.show()
