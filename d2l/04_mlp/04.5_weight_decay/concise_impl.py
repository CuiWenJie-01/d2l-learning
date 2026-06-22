import torch
from torch import nn
from d2l import torch as d2l

# ================= 1. 初始化超参数与生成数据集 =================
# 为了使过拟合的效果更加明显，我们将问题的维数增加到 d = 200
# 并使用一个只包含 20 个样本的小训练集
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

# 构造人工数据集（包含高斯噪声）
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# ================= 2. 评估损失的辅助函数 =================
def evaluate_loss(net, data_iter, loss):
    """评估模型在给定数据集上的平均损失"""
    net.eval()  # 设置为评估模式
    metric = d2l.Accumulator(2)  # 损失的总和, 样本数量
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# ================= 3. 简洁实现的核心训练函数 =================
def train_concise(wd):
    """使用 PyTorch 自带的 weight_decay 参数实现权重衰减"""
    # 3.1 定义单层线性网络
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
        
    # 3.2 定义标准均方误差损失
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    
    # 3.3 关键点：在优化器中为权重 w 配置 weight_decay，而偏置 b 不衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}
    ], lr=lr)
    
    # 3.4 初始化动图绘制器
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    
    # 3.5 标准的 PyTorch 训练循环
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            # 因为 reduction='none'，返回的是小批量向量，需通过 mean() 转化为标量再反向传播
            l.mean().backward()  
            trainer.step()
            
        # 每隔 5 个 epoch 记录并绘制一次损失
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (
                evaluate_loss(net, train_iter, loss), 
                evaluate_loss(net, test_iter, loss)
            ))
            
    print(f'当 weight_decay={wd} 时，学习到的 w 的 L2 范数：', net[0].weight.norm().item())

# ================= 4. 执行对比实验 =================
if __name__ == '__main__':
    # 实验 1：不使用权重衰减 (wd=0)，观察严重的过拟合现象
    print("====== 1. 忽略正则化直接训练 (wd=0) ======")
    train_concise(0)
    
    # 实验 2：使用权重衰减 (wd=3)，观察测试集损失的下降，过拟合得到缓解
    print("\n====== 2. 使用权重衰减训练 (wd=3) ======")
    train_concise(3)