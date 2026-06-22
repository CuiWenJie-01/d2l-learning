import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# ================= 1. 生成数据集 =================
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 真实权重

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]
]

# ================= 2. 替代 d2l.train_epoch_ch3 的本地实现 =================
def train_epoch_local(net, train_iter, loss, updater):
    """训练模型一个迭代周期（自定义本地实现）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置优化器
            updater.zero_grad()
            l.mean().backward()  # 注意：由于 reduction='none'，这里需要取 mean 后再反向传播
            updater.step()
        else:
            # 使用 D2L 自定义优化器
            l.sum().backward()
            updater(X.shape[0])

def evaluate_loss(net, data_iter, loss):
    """评估模型在给定数据集上的损失"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# ================= 3. 修改后的训练主函数 =================
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    
    for epoch in range(num_epochs):
        # ⚠️ 这里替换掉了原先报错的 d2l.train_epoch_ch3
        train_epoch_local(net, train_iter, loss, trainer)
        
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
            
    print('学习到的模型参数权重 (weight):')
    print(net[0].weight.data.numpy())

# ================= 4. 执行实验 =================
if __name__ == '__main__':
    print("====== 1. 正常拟合 (三阶多项式特征) ======")
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    
    print("\n====== 2. 欠拟合 (线性特征) ======")
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    
    print("\n====== 3. 过拟合 (二十阶多项式特征) ======")
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)