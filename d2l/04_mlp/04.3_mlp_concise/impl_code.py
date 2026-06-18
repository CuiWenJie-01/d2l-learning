import torch
from torch import nn
from d2l import torch as d2l

# 1. 从你刚写好的本地工具库中导入训练函数
from my_d2l_utils import train_ch3

# 2. 定义模型架构
net = nn.Sequential(
    nn.Flatten(),

    nn.Linear(784, 256),# 隐藏层 1：输入784，输出256
    nn.ReLU(),# 隐藏层 1 的激活函数

    nn.Linear(256,64),# 隐藏层 2：输入256，输出64
    nn.ReLU(),# 隐藏层 2 的激活函数

    nn.Linear(64, 10)# 输出层：输入64，输出10
)

# 3. 初始化模型权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) # 正态分布初始化

net.apply(init_weights)

# 4. 设置超参数、损失函数和优化器
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 5. 加载数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 6. 调用你抽离出来的训练函数
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 强制弹出折线图窗口
d2l.plt.show()