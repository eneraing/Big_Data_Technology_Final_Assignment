import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# ===============================
# 1. 模拟实时数据流（含缺失和异常）
# ===============================
def generate_data_stream(
    length=500,
    mu=50,
    sigma=10,
    miss_prob=0.1,
    anomaly_prob=0.05
):
    true_data = []
    observed_data = []

    for _ in range(length):
        x = np.random.normal(mu, sigma)
        true_data.append(x)

        r = random.random()
        if r < miss_prob:
            observed_data.append(None)          # 缺失
        elif r < miss_prob + anomaly_prob:
            observed_data.append(x * 6)         # 强异常
        else:
            observed_data.append(x)

    return np.array(true_data), observed_data


# ===============================
# 2. 滑动窗口统计工具
# ===============================
class SlidingWindow:
    def __init__(self, size=20):
        self.window = deque(maxlen=size)

    def update(self, value):
        if value is not None:
            self.window.append(value)

    def mean(self):
        return np.mean(self.window) if len(self.window) > 0 else 0.0

    def std(self):
        return np.std(self.window) if len(self.window) > 1 else 1.0


# ===============================
# 3. 固定均值清洗策略
# ===============================
def clean_fixed_mean(observed):
    cleaned = []
    history = []

    for x in observed:
        if x is None:
            fill = np.mean(history) if history else 0.0
            cleaned.append(fill)
        else:
            cleaned.append(x)
            history.append(x)

    return np.array(cleaned)


# ===============================
# 4. 固定滑动窗口清洗策略
# ===============================
def clean_fixed_window(observed, window_size=20):
    cleaned = []
    window = SlidingWindow(window_size)

    for x in observed:
        if x is None:
            cleaned.append(window.mean())
        else:
            window.update(x)
            cleaned.append(x)

    return np.array(cleaned)


# ===============================
# 5. 自适应清洗策略（核心创新）
# ===============================
def clean_adaptive(
    observed,
    window_size=20,
    z_threshold=3.0,
    anomaly_rate_threshold=0.3
):
    cleaned = []
    window = SlidingWindow(window_size)
    anomaly_history = []
    strategy_trace = []

    for x in observed:
        # 缺失值处理
        if x is None:
            value = window.mean()
            strategy_trace.append("mean_fill")
        else:
            mu = window.mean()
            std = window.std()
            z = abs(x - mu) / std if std > 0 else 0.0
            is_anomaly = z > z_threshold
            anomaly_history.append(is_anomaly)

            recent = anomaly_history[-window_size:]
            anomaly_rate = sum(recent) / len(recent) if recent else 0.0

            # 自适应策略决策
            if anomaly_rate > anomaly_rate_threshold:
                value = window.mean()      # 鲁棒修复
                strategy_trace.append("robust")
            else:
                value = x                  # 保留原值
                strategy_trace.append("raw")

        window.update(value)
        cleaned.append(value)

    return np.array(cleaned), strategy_trace


# ===============================
# 6. 评价指标
# ===============================
def mean_shift(true_data, cleaned_data):
    return abs(np.mean(cleaned_data) - np.mean(true_data))

def variance_shift(true_data, cleaned_data):
    return abs(np.var(cleaned_data) - np.var(true_data))


# ===============================
# 7. 主实验流程
# ===============================
def main():
    # 生成数据
    true_data, observed_data = generate_data_stream()

    # 三种清洗策略
    fixed_mean_data = clean_fixed_mean(observed_data)
    fixed_window_data = clean_fixed_window(observed_data)
    adaptive_data, strategy_trace = clean_adaptive(observed_data)

    # ===============================
    # 图 1：原始数据流
    # ===============================
    plt.figure()
    plt.plot(true_data, label="True Data")
    plt.plot(
        [x if x is not None else np.nan for x in observed_data],
        linestyle="dotted",
        label="Observed Data"
    )
    plt.title("Raw Data Stream with Missing and Anomalies")
    plt.legend()
    plt.show()

    # ===============================
    # 图 2：清洗策略对比（核心图）
    # ===============================
    plt.figure()
    plt.plot(fixed_mean_data, label="Fixed Mean")
    plt.plot(fixed_window_data, label="Fixed Window")
    plt.plot(adaptive_data, label="Adaptive Strategy")
    plt.title("Comparison of Cleaning Strategies")
    plt.legend()
    plt.show()

    # ===============================
    # 图 3：均值偏移对比
    # ===============================
    methods = ["Fixed Mean", "Fixed Window", "Adaptive"]
    mean_shifts = [
        mean_shift(true_data, fixed_mean_data),
        mean_shift(true_data, fixed_window_data),
        mean_shift(true_data, adaptive_data)
    ]

    plt.figure()
    plt.bar(methods, mean_shifts)
    plt.title("Mean Shift after Cleaning")
    plt.ylabel("Absolute Mean Shift")
    plt.show()

    # ===============================
    # 图 4：方差偏移对比
    # ===============================
    var_shifts = [
        variance_shift(true_data, fixed_mean_data),
        variance_shift(true_data, fixed_window_data),
        variance_shift(true_data, adaptive_data)
    ]

    plt.figure()
    plt.bar(methods, var_shifts)
    plt.title("Variance Shift after Cleaning")
    plt.ylabel("Absolute Variance Shift")
    plt.show()

    # ===============================
    # 图 5：自适应策略切换行为
    # ===============================
    strategy_numeric = [
        0 if s == "raw" else 1 for s in strategy_trace
    ]

    plt.figure()
    plt.plot(strategy_numeric)
    plt.yticks([0, 1], ["Raw", "Robust/Fill"])
    plt.title("Adaptive Strategy Switching Behavior")
    plt.xlabel("Time Step")
    plt.show()


# ===============================
# 8. 运行
# ===============================
if __name__ == "__main__":
    main()
