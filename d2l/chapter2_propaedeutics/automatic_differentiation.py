# 1.为什么计算二阶导数比一阶导数的开销要更大？
'''
计算一阶导数时，反向传播只需从输出到输入遍历一次计算图，并存储中间梯度。而计算二阶导数（如 Hessian 矩阵或向量积）需要：

先完成一阶导数的反向传播，保留计算图中的中间变量（或重新计算）；

再对一阶导数结果进行第二次反向传播，这意味着需要额外存储一阶导数的计算图，或使用自动微分技术（如 PyTorch 的 create_graph=True）来构建高阶导数图。

内存和计算量均大致翻倍，且对于高维输出，二阶张量（Hessian）的存储和计算成本为平方级。

因此，二阶导数的开销显著大于一阶导数。
'''
# 2.在运行反向传播函数之后，立即再次运行它，看看会发生什么。
# import torch

# x = torch.tensor([2.0], requires_grad=True)
# y = x ** 2

# # 第一次反向传播
# # y.backward(retain_graph=True)  # 保留计算图
# y.backward()
# print(f"第一次梯度: {x.grad}")  # 输出 4.0

# # 第二次反向传播，梯度累加
# # y.backward(retain_graph=True)
# y.backward()
# print(f"第二次后梯度: {x.grad}")  # 输出 8.0 = 4+4

# 若不保留计算图，第二次调用会报错
# y.backward()  # RuntimeError: Trying to backward through the graph a second time

# 3.在控制流的例子中，我们计算 d 关于 a 的导数，如果将变量 a 更改为随机向量或矩阵，会发生什么？
# import torch

# def control_flow_func(a):
#     # 模拟控制流：若元素平方和大于阈值，则乘以2，否则加1
#     if a.sum() > 10:
#         return 2 * a.sum()
#     else:
#         return a.sum() + 1

# a = torch.randn(3, 4, requires_grad=True)
# print(a)
# print(a.sum())
# d = control_flow_func(a)
# d.backward()
# print(a.grad)  # 梯度与 a 同形状，值取决于 a.sum() 是否 >10

# 4. 重新设计一个求控制梯度的例子，运行并分析结果。
# import torch

# def custom_func(x):
#     if x[0] > 0:
#         return (x.sum()) ** 2
#     else:
#         return (x.sum()) ** 3

# # 情况1：x[0] > 0
# x1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y1 = custom_func(x1)
# y1.backward()
# print("x[0]>0 时的梯度:", x1.grad)  # 梯度 = 2*sum(x) * 1 = 2*6=12 对每个分量

# # 重置梯度
# x1.grad.zero_()

# # 情况2：x[0] <= 0
# x2 = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
# y2 = custom_func(x2)
# y2.backward()
# print("x[0]<=0 时的梯度:", x2.grad)  # 梯度 = 3*(sum(x))^2 * 1 = 3*4^2=48

# print(x1)
'''
分析：梯度值在不同控制流分支下不同，且所有分量梯度相等（因为函数内部是标量 sum(x) 的幂）。这体现了控制流对梯度计算的直接影响。
'''

# 5. 使 f(x) = sin(x)，绘制 f(x) 和 df(x)/dx 的图像，其中后者不使用 f'(x)=cos(x)
import torch
import matplotlib.pyplot as plt
import numpy as np

# 定义 x 的取值范围
x_np = np.linspace(-np.pi, np.pi, 200)
grads = [] # 用于存储梯度值

for x_val in x_np:
    # 每次创建一个新的标量叶子张量
    x = torch.tensor(x_val, requires_grad=True)
    f = torch.sin(x)
    f.backward()
    grads.append(x.grad.item())  # 现在 x.grad 一定存在且正确
# 在计算 grads 之后，选取几个点打印
test_x = torch.tensor([-0.91], requires_grad=True)
test_f = torch.sin(test_x)
test_f.backward()
print(f"x=-0.91: 自动微分导数 = {test_x.grad.item():.6f}")
print(f"理论 cos(x) = {torch.cos(torch.tensor(-0.91)).item():.6f}")

# 计算 f(x) 的值用于绘图（也可以直接用 torch.sin，但为保持数值一致，同上方式）
x_vals = torch.tensor(x_np, requires_grad=False)
f_vals = torch.sin(x_vals).numpy()

# 绘图
plt.plot(x_np, f_vals, label='f(x)=sin(x)')
plt.plot(x_np, grads, '--', label="df/dx (auto diff)")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()