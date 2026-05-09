# 1. 样本均值的分布模拟（改变 m 和 n）
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 总体分布：均匀分布 U(0,1)
# 理论均值 = 0.5，理论方差 = 1/12 ≈ 0.0833

def simulate_sample_means(m, n, dist='uniform'):
    """进行 m 组实验，每组抽取 n 个样本，返回每组样本的均值"""
    if dist == 'uniform':
        samples = np.random.uniform(0, 1, size=(m, n))
    elif dist == 'exponential':
        samples = np.random.exponential(1, size=(m, n))
    else:
        raise ValueError("不支持该分布")
    return np.mean(samples, axis=1)

# 不同参数组合
params = [(500, 10), (500, 50), (2000, 10), (2000, 50)]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (m, n) in enumerate(params):
    means = simulate_sample_means(m, n, 'uniform')
    ax = axes[idx]
    ax.hist(means, bins=30, density=True, alpha=0.7, edgecolor='black')
    # 理论正态分布：均值=0.5，标准差=sqrt(1/12 / n)
    theo_mean = 0.5
    theo_std = np.sqrt(1/12 / n)
    x = np.linspace(theo_mean - 4*theo_std, theo_mean + 4*theo_std, 200)
    ax.plot(x, stats.norm.pdf(x, theo_mean, theo_std), 'r-', lw=2, label='理论正态分布')
    ax.set_title(f'm={m}, n={n}')
    ax.set_xlabel('样本均值')
    ax.set_ylabel('密度')
    ax.legend()
plt.tight_layout()
plt.show()