import random
import torch
from d2l import torch as d2l

# 生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) # 生成X
    y = torch.matmul(X, w) + b # 计算y=Xw+b
    y += torch.normal(0, 0.01, y.shape) # 添加噪声
    return X, y.reshape((-1, 1)) # 返回特征矩阵X和重塑后的标签向量y，y被重塑为列向量（-1表示自动推断维度大小）。

true_w = torch.tensor([2, -3.4]) # 生成真实权重参数
true_b = 4.2 # 真实偏置参数
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)# 数据集的大小
    indices = list(range(num_examples)) # 生成一个索引列表
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor( # 生成一个索引列表
            indices[i: min(i + batch_size, num_examples)]) # 最小值确保最后一个批次不会超出数据集的范围
        yield features[batch_indices], labels[batch_indices] # 返回特征矩阵和标签向量

batch_size = 10 # 批量大小

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # 生成权重参数
b = torch.zeros(1, requires_grad=True) # 生成偏置参数

# 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
# 参数、学习率、批量大小
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 梯度除以批量大小
            param.grad.zero_() # 将参数的梯度清零，以便下一次迭代计算新的梯度

# 训练模型
lr = 0.03
num_epochs = 3 # 迭代轮数
net = linreg # 线性回归的模型
loss = squared_loss # 均方损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():# 不记录梯度
        train_l = loss(net(features, w, b), labels) # 训练损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')