import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# 1. 数据生成
# ===============================
def generate_data(length=500, anomaly_prob=0.1, miss_prob=0.1):
    true, obs = [], []
    for _ in range(length):
        x = np.random.normal(50, 10)
        true.append(x)

        r = random.random()
        if r < miss_prob:
            obs.append(None)
        elif r < miss_prob + anomaly_prob:
            obs.append(x * 6)
        else:
            obs.append(x)
    return np.array(true), obs

# ===============================
# 2. 清洗策略
# ===============================
def fixed_mean(obs):
    vals = [x for x in obs if x is not None]
    m = np.mean(vals)
    return np.array([m if x is None else x for x in obs]), len(obs)

def sliding_window(obs, w=20):
    out, repair = [], 0
    for i, x in enumerate(obs):
        hist = [v for v in out[max(0, i-w):] if v is not None]
        if x is None:
            out.append(np.mean(hist) if hist else 0)
            repair += 1
        else:
            out.append(x)
    return np.array(out), repair

def adaptive(obs, w=20, z_th=3):
    out, repair = [], 0
    for i, x in enumerate(obs):
        if i < w or x is None:
            val = np.mean(out) if out else 0
            repair += 1
        else:
            mu = np.mean(out[i-w:i])
            std = np.std(out[i-w:i]) + 1e-6
            if abs(x - mu) / std > z_th:
                val = mu
                repair += 1
            else:
                val = x
        out.append(val)
    return np.array(out), repair

# ===============================
# 3. 评价指标
# ===============================
def evaluate(true, cleaned, repair_cnt):
    return {
        "Mean Shift": abs(np.mean(cleaned) - np.mean(true)),
        "Variance Shift": abs(np.var(cleaned) - np.var(true)),
        "RMSE": np.sqrt(np.mean((cleaned - true) ** 2)),
        "Repair Ratio": repair_cnt / len(true)
    }

# ===============================
# 4. 主实验
# ===============================
true, obs = generate_data()

res = {}
cleaned, r = fixed_mean(obs)
res["Fixed Mean"] = evaluate(true, cleaned, r)

cleaned, r = sliding_window(obs)
res["Sliding Window"] = evaluate(true, cleaned, r)

cleaned, r = adaptive(obs)
res["Adaptive"] = evaluate(true, cleaned, r)

df = pd.DataFrame(res).T
print("\n=== Evaluation Table ===\n")
print(df)

# ===============================
# 5. Figure 1：柱状图
# ===============================
df.plot(kind="bar")
plt.title("Comparison under Multiple Data Quality Metrics")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ===============================
# 6. Figure 2：雷达图
# ===============================
labels = df.columns
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure()
ax = plt.subplot(111, polar=True)

for idx, row in df.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=idx)
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Radar Chart of Cleaning Performance")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()
