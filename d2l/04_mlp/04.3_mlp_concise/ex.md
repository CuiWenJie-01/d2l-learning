针对[4.3. 多层感知机的简洁实现](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html)课后练习题，这里为你提供详细的代码修改方案与实验结论解答：

---

### 1. 尝试添加不同数量的隐藏层（也可以修改学习率），怎么样设置效果最好？

**【解答与修改方案】**
原模型只有 1 个隐藏层（784 -> 256 -> 10）。我们可以通过在 `nn.Sequential` 中堆叠更多的 `nn.Linear` 和 `nn.ReLU` 来增加隐藏层数量。

根据社区和通用实验经验，对于 Fashion-MNIST 这种简单的图像数据集，**盲目增加层数（比如加到 4、5 层）容易导致过拟合或梯度消失，如果不配合延长 Epoch，准确率反而会下降**。

* **较佳的设置方案**：添加至 **2 个隐藏层**，适当将架构加宽、加深，并微调学习率与 Epoch。例如：`784 -> 512 -> 128 -> 10`，同时将 `lr` 设为 `0.2` 或 `0.3`，`num_epochs` 提高到 `20`。

**代码修改示例（2个隐藏层）：**

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),  # 隐藏层 1：扩充到 512 节点
    nn.ReLU(),
    nn.Linear(512, 128),  # 隐藏层 2：降维到 128 节点
    nn.ReLU(),
    nn.Linear(128, 10)    # 输出层
)

# 配合调整超参数
batch_size, lr, num_epochs = 256, 0.2, 20  # 稍微提高学习率并延长训练轮数

```

---

### 2. 尝试不同的激活函数，哪个效果最好？

**【解答与修改方案】**
PyTorch 提供了多种激活函数，常见的有 `nn.ReLU()`、`nn.Tanh()` 和 `nn.Sigmoid()`。你只需要替换 `nn.Sequential` 里的激活层即可。

**代码修改示例：**

```python
# 测试 Tanh 激活函数
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Tanh(),  # 将 nn.ReLU() 替换为 nn.Tanh()
    nn.Linear(256, 10)
)

```

**【实验结论对比】**

* **`nn.ReLU()`（最好）**：收敛速度最快，最终准确率最高。因为 ReLU 在正区间的梯度恒为 1，能有效缓解梯度消失。
* **`nn.Tanh()`（次之）**：效果尚可，曲线也比较平滑，但在输入过大或过小时梯度趋近于 0，收敛速度慢于 ReLU。
* **`nn.Sigmoid()`（最差）**：在当前的超参数（`lr=0.1`, `num_epochs=10`）下**效果非常糟糕**，甚至可能导致模型不收敛（准确率停留在 0.1 左右）。这是因为 Sigmoid 极易发生梯度消失，需要更多的迭代轮数（Epoch）或更精准的权重初始化配合。

---

### 3. 尝试不同的方案来初始化权重，什么方法效果最好？

**【解答与修改方案】**
原代码使用的是正态分布初始化 `nn.init.normal_(m.weight, std=0.01)`。我们可以尝试改为 **Xavier 初始化**（常用于 Tanh/Sigmoid）或 **Kaiming 初始化**（也叫 He 初始化，常用于 ReLU），或者直接用**均匀分布**。

我们可以通过修改 `init_weights` 函数来进行对比：

```python
def init_weights(m):
    if type(m) == nn.Linear:
        # 方案 A: Kaiming 初始化（配合 ReLU 效果最好）
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        # 方案 B: Xavier 初始化（配合 Tanh 效果好）
        # nn.init.xavier_normal_(m.weight)
        
        # 方案 C: 均匀分布初始化（注意：区间一定要关于y轴对称，如 -1 到 1）
        # nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        
        # 别忘了初始化偏置（Bias），通常初始化为 0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

```

**【实验结论对比】**

* **`nn.init.kaiming_normal_`（配合 ReLU 效果最好）**：它是专门为 ReLU 及其变体设计的初始化方法，能够很好地保持前向传播和反向传播时的方差稳定，训练曲线最稳健，不容易崩。
* **原书的 `normal_(std=0.01)**`：在这个简单网络里效果也不错，但如果网络加深，手动指定的 `std=0.01` 就会失效。
* **`nn.init.uniform_`（均匀分布）**：如果区间设置不对称（比如错误地设为默认的 `[0, 1]`），会导致激活值全为正数，进而引发梯度消失，模型完全无法训练。