# 第1题
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义函数
# x = np.linspace(0.2, 2, 400) # 生成400个点
# y = x**3 - 1/x

# # 切线
# tangent = 4*x - 4

# plt.plot(x, y, label='f(x)=x^3-1/x')
# plt.plot(x, tangent, label='tangent line')

# # 切点
# plt.scatter([1], [0]) # 画点

# plt.legend()
# plt.grid()
# plt.show()

# 第2题
# import torch

# x = torch.tensor(2.0, requires_grad=True) # 定义一个张量x，要求导数

# f = 3 * x**2 + 5 * torch.exp(x) # 定义一个函数f(x)

# '''
# backward() - 从输出节点（f）反向遍历计算图，应用链式法则计算梯度,通过 反向传播算法 自动计算梯度
# '''
# f.backward() # 计算函数f(x)关于x的梯度

# print(x.grad) # x.grad存储计算得到的梯度值

# 第3题
# import torch

# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# print(x)
# f = torch.sum(x**2)

# f.backward()


# print(x.grad)

# 第4题
import torch

a = torch.tensor(1.0, requires_grad=True) # 定义一个张量a，要求导数,1.0为常数
b = torch.tensor(2.0, requires_grad=True) # 定义一个张量b，要求导数,2.0为常数

x = a + b
y = a * b
z = a**2 + b**2

u = x**2 + y + z

u.backward()

print("du/da =", a.grad)
print("du/db =", b.grad)

'''
| 题目  | 核心知识    |
| --- | ------- |
| 第1题 | 导数、切线   |
| 第2题 | 梯度、一元求导 |
| 第3题 | 向量梯度    |
| 第4题 | 多变量链式法则 |

这几题本质上是在为后面的：
    自动求导（Autograd）
    反向传播（Backpropagation）
    深度学习梯度下降
'''