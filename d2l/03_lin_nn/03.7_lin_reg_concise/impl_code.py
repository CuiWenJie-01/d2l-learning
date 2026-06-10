import torch
from torch import nn
from d2l import torch as d2l


def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 定义显示图像的函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 显示训练集中的一些图像
    X, y = next(iter(train_iter))
    # 取出第一个批次的一部分图像进行展示
    show_images(X[:18].reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y[:18]))
    d2l.plt.show()

    # 初始化模型参数
    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);

    # 重新审视Softmax的实现(按照上一节课后习题的方法改进Softmax的实现)
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    num_epochs = 10

    # 手动训练循环
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_acc = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        
        # 记录数据
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    # 绘制图表
    d2l.set_figsize()
    d2l.plot(range(1, num_epochs + 1), [train_loss_history, train_acc_history, test_acc_history],
             xlabel='epoch', ylabel='loss/acc',
             legend=['train loss', 'train acc', 'test acc'],
             xlim=[1, num_epochs], ylim=[0, 1])
    d2l.plt.show()

if __name__ == '__main__':
    main()