import numpy as np
import random
import matplotlib.pyplot as plt

# ===============================
# 数据生成
# ===============================
def generate_data(length=500, anomaly_prob=0.05):
    true, obs = [], []
    for _ in range(length):
        x = np.random.normal(50, 10)
        true.append(x)
        if random.random() < anomaly_prob:
            obs.append(x * 6)
        else:
            obs.append(x)
    return np.array(true), np.array(obs)

# ===============================
# 清洗策略
# ===============================
def fixed_mean(obs):
    return np.full_like(obs, np.mean(obs))

def sliding_window(obs, w=20):
    out = []
    for i in range(len(obs)):
        start = max(0, i - w)
        out.append(np.mean(obs[start:i+1]))
    return np.array(out)

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

# ===============================
# 实验
# ===============================
ratios = [0.01, 0.05, 0.1, 0.2]
m1, m2, m3 = [], [], []

for r in ratios:
    true, obs = generate_data(anomaly_prob=r)
    m1.append(abs(np.mean(fixed_mean(obs)) - np.mean(true)))
    m2.append(abs(np.mean(sliding_window(obs)) - np.mean(true)))
    m3.append(abs(np.mean(adaptive(obs)) - np.mean(true)))

plt.figure()
plt.plot(ratios, m1, marker='o', label="Fixed Mean")
plt.plot(ratios, m2, marker='o', label="Sliding Window")
plt.plot(ratios, m3, marker='o', label="Adaptive")
plt.xlabel("Anomaly Ratio")
plt.ylabel("Mean Shift")
plt.title("Robustness under Increasing Anomaly Ratio")
plt.legend()
plt.show()
