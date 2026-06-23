import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ================= 1. 初始化超参数与加载数据 =================
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.5, 0.2
num_epochs, lr, batch_size = 10, 0.5, 256

# 加载 Fashion-MNIST 数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# ================= 2. 简洁实现：定义网络架构 =================
# 在高级 API 中，我们只需在对应的全连接层（Linear）和激活函数（ReLU）之后，
# 插入 nn.Dropout 层并传入暂退概率即可。
net = nn.Sequential(
    nn.Flatten(), # 展平层
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第一个全连接层之后添加一个 dropout 层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第二个全连接层之后添加一个 dropout 层
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)

# 初始化模型权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

net.apply(init_weights)

# ================= 3. 评估指标辅助函数 =================
def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 💡 关键：测试模式下关闭 Dropout
    metric = d2l.Accumulator(2)  # 正确预测数, 总预测数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# ================= 4. 标准的 PyTorch 训练循环 =================
def train_dropout(net, train_iter, test_iter, num_epochs, lr):
    # 采用标准交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 初始化动图绘制器
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.5, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        net.train()  # 💡 关键：训练模式下开启 Dropout
        metric = d2l.Accumulator(3)  # 训练损失总和, 训练准确度总和, 样本数
        
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()  # 显式转为标量求导，规避不必要报错
            trainer.step()
            
            metric.add(l.sum().item(), d2l.accuracy(y_hat, y), y.numel())
            
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_acc = evaluate_accuracy(net, test_iter)
        
        # 记录并绘制当前 epoch 的结果
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
    print(f'训练完成！最终 Train Loss: {train_metrics[0]:.4f}, '
          f'Train Acc: {train_metrics[1]:.4f}, Test Acc: {test_acc:.4f}')

# ================= 5. 启动训练 =================
if __name__ == '__main__':
    print("====== 开始运行 4.6.5 暂退法简洁实现 ======")
    train_dropout(net, train_iter, test_iter, num_epochs, lr)
    plt.show()