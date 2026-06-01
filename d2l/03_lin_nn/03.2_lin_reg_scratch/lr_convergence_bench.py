import torch
from d2l import torch as d2l

# 生成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 初始化参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 学习率列表
lrs = [0.001, 0.03, 0.1, 1]

def train(lr):
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(10):
        y_hat = torch.matmul(features, w) + b
        loss = ((y_hat - labels.reshape(y_hat.shape)) ** 2 / 2).mean()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        print(f"lr={lr}, epoch={epoch}, loss={loss.item():.4f}")

for lr in lrs:
    print("\n====================")
    train(lr)