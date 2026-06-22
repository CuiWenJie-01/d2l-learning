### 1. 在本节的估计问题中使用 $\lambda$ 的值进行实验。绘制训练和测试精度关于 $\lambda$ 的函数。观察到了什么？

**【解答】**
当 $\lambda$（代码中的 `wd`）取不同值时，模型展现出不同的状态：

* **$\lambda$ 过小（如 0）**：模型没有受到正则化约束。由于训练样本极少（20个）且特征维度高（200维），模型会完美拟合训练集的所有噪声，导致 **Train Loss 极低，但 Test Loss 极大**（严重过拟合）。
* **$\lambda$ 适中（如 3 到 10 之间）**：随着 $\lambda$ 的增大，对高权重的惩罚生效，限制了模型容量。**Train Loss 略微上升，但 Test Loss 显著下降**，泛化性能达到最佳。
* **$\lambda$ 过大（如 >50 或 100）**：参数被强行压制迫近于 0，模型失去了拟合数据的能力。此时 **Train Loss 和 Test Loss 都会变得非常大**（欠拟合），且由于惩罚过重，训练曲线可能会出现剧烈的震荡。

**核心实验绘图代码思路**：
你可以通过一个外层循环遍历不同的 `wd`，记录最终的 loss 并使用 `d2l.plot` 绘制：

```python
wds = list(range(0, 30, 2))
train_losses, test_losses = [], []
for wd in wds:
    # 这里的 train_concise_return 需要修改原函数使其返回最终的评估 loss
    tr_l, te_l = train_concise_return(wd) 
    train_losses.append(tr_l)
    test_losses.append(te_l)
d2l.plot(wds, [train_losses, test_losses], 'lambda', 'loss', legend=['train', 'test'])

```

---

### 2. 使用验证集来找到最佳值 $\lambda$。它真的是最优值吗？这有关系吗？

**【解答】**

1. **它不是绝对意义上的最优值。** 因为验证集只是从未知的数据真实分布中采样出来的一个子集。我们在当前验证集上选出的“最佳 $\lambda$”，只是对该特定验证集最优。如果换一组测试数据，这个 $\lambda$ 可能就不是绝对最优了。
2. **但这没有关系。** 机器学习的核心目标是追求**泛化能力**（在未知数据上的表现良好），而不是追求极致的绝对数学最值。只要验证集足够具有代表性（与测试集独立同分布），通过它选出来的 $\lambda$ 就能有效缓解过拟合，这就足以满足工业和工程应用的需求。

---

### 3. 如果我们使用 $\sum_i |w_i|$ 作为我们选择的惩罚（$L_1$ 正则化），那么更新方程会是什么样子？

**【解答】**
$L_1$ 正则化对应的损失函数为：


$$L(\mathbf{w}, b) + \lambda \sum_i |w_i|$$

对权重 $w_i$ 求导数，由于 $|w_i|$ 在 0 处不可导，我们使用次导数（Subderivative）。$|w_i|$ 的导数是符号函数 $\text{sign}(w_i)$：


$$\text{sign}(w_i) = \begin{cases} 1 & w_i > 0 \\ 0 & w_i = 0 \\ -1 & w_i < 0 \end{cases}$$

因此，使用小批量随机梯度下降更新权重时的**更新方程**为：


$$\mathbf{w} \leftarrow \mathbf{w} - \eta \lambda \text{sign}(\mathbf{w}) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$$

**对比与物理意义**：

* $L_2$ 正则化（权重衰减）每次更新时是将权重**按比例缩小**（乘以 $1 - \eta\lambda$）。
* $L_1$ 正则化每次更新时是将权重**减去一个固定常数 $\eta\lambda$**（向 0 靠拢）。这会导致不重要的权重直接被强制抹为 0，从而产生**稀疏解**，因此 $L_1$ 常用于特征选择。

---

### 4. 我们知道 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$。能找到类似的矩阵方程吗（见 Frobenius 范数）？

**【解答】**
对于矩阵 $\mathbf{X}$，其 Frobenius 范数的平方 $\|\mathbf{X}\|_F^2$ 是矩阵所有元素的平方和。
它的矩阵方程形式可以通过矩阵的迹（Trace）和矩阵乘法来表示：


$$\|\mathbf{X}\|_F^2 = \text{trace}(\mathbf{X}^\top \mathbf{X})$$

或者等价地写成：


$$\|\mathbf{X}\|_F^2 = \text{trace}(\mathbf{X} \mathbf{X}^\top)$$


*注：$\text{trace}(\mathbf{A})$ 表示矩阵 $\mathbf{A}$ 的主对角线元素之和。*

---

### 5. 除了权重衰减、增加训练数据、使用适当复杂度的模型之外，还能想出其他什么方法来处理过拟合？

**【解答】**
深度学习中有非常丰富的正则化手段来对抗过拟合，常见的有：

1. **暂退法（Dropout）**：在训练期间随机丢弃一部分神经元，迫使网络不依赖某些特定神经元，这也是下一节（4.6 节）的核心内容。
2. **提前停止（Early Stopping）**：在训练过程中评估验证集损失。当发现验证集损失不再下降甚至开始上升时，即使训练未完成也立即停止训练。
3. **数据增广（Data Augmentation）**：通过对原始数据进行旋转、裁剪、平移、加入噪声等操作，人为扩大数据集。
4. **批量归一化（Batch Normalization）**：虽然主要用于加速收敛，但由于每个 Batch 的均值和方差略有不同，它引入了微小的噪声，起到了一定的正则化效果。
5. **标签平滑（Label Smoothing）**：不让模型过于自信地预测 100% 的概率，给硬标签加入少许模糊度，防止参数过大。

---

### 6. 在贝叶斯统计中，如何得到带正则化的 $P(\mathbf{w})$？

**【解答】**
根据贝叶斯公式，后验概率公式为：$P(\mathbf{w} \mid \mathbf{x}) \propto P(\mathbf{x} \mid \mathbf{w}) P(\mathbf{w})$。
对公式两边取负对数 $-\ln$，可以将最大后验概率（MAP）问题转化为最小化目标函数：


$$\text{argmin} \left( -\ln P(\mathbf{x} \mid \mathbf{w}) - \ln P(\mathbf{w}) \right)$$


其中 $-\ln P(\mathbf{x} \mid \mathbf{w})$ 对应我们的传统预测损失（如均方误差），而 **$-\ln P(\mathbf{w})$ 则直接对应正则化惩罚项**。

通过假设参数 $\mathbf{w}$ 具有不同的**先验概率分布（Prior Distribution）**，就可以推导出不同的正则化项：

* **$L_2$ 正则化（权重衰减）**：假设 $\mathbf{w}$ 的各个分量服从独立同分布的**高斯（正态）分布** $\mathcal{N}(0, \sigma^2)$。其概率密度函数的负对数展开后，会包含 $\frac{1}{2\sigma^2} \|\mathbf{w}\|^2$，这与 $L_2$ 正则化完全等价。其中 $\lambda = \frac{1}{\sigma^2}$。
* **$L_1$ 正则化**：假设 $\mathbf{w}$ 服从均值为 0 的**拉普拉斯分布（Laplace Distribution）** $P(w_i) = \frac{1}{2b} \exp\left(-\frac{|w_i|}{b}\right)$。其负对数展开后包含 $\frac{1}{b} \sum |w_i|$，这与 $L_1$ 正则化完全等价。