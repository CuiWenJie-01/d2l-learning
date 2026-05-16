import time
import numpy as np
import torch

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.perf_counter()   # 使用更高精度的计时器

    def stop(self):
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

# ---------- 关键：将维度放大 100 倍 ----------
n = 1_000_000          # 100万个元素
a = torch.ones(n)
b = torch.ones(n)

# 方法1：Python for 循环（只跑少量次以节省时间）
c = torch.zeros(n)
timer = Timer()
# 对百万次循环会很慢，这里只测速，慎重运行
for i in range(n):
    c[i] = a[i] + b[i]
time_loop = timer.stop()
print(f"for 循环耗时: {time_loop:.6f} 秒")

# 方法2：矢量化加法
timer.start()
d = a + b
time_vec = timer.stop()
print(f"矢量化加法耗时: {time_vec:.6f} 秒")

# 计算加速比
if time_vec > 0:
    print(f"矢量化比 for 循环快约 {time_loop / time_vec:.0f} 倍")
else:
    print("矢量化耗时极短（近乎0），加速比可达数千甚至数万倍")

print("两种方法结果一致:", torch.allclose(c, d))