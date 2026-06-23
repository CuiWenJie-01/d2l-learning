### 1. 假设一些标量函数 $f$ 的输入 $\mathbf{X}$ 是 $n \times m$ 矩阵。$f$ 相对于 $\mathbf{X}$ 的梯度维数是多少？

**【解答】**
梯度的维数与输入矩阵 $\mathbf{X}$ 的维数**完全相同**，即 **$n \times m$**。
因为标量对矩阵求导，其本质是函数 $f$ 对矩阵 $\mathbf{X}$ 中的每一个元素 $X_{ij}$ 分别求偏导数，并排列成与原矩阵相同形状的矩阵：


$$\frac{\partial f}{\partial \mathbf{X}} = \begin{bmatrix} \frac{\partial f}{\partial X_{11}} & \dots & \frac{\partial f}{\partial X_{1m}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial X_{n1}} & \dots & \frac{\partial f}{\partial X_{nm}} \end{bmatrix}_{n \times m}$$

---

### 2. 向本节中描述的模型的隐藏层添加偏置项（不需要在正则化项中包含偏置项）。

#### (1) 画出相应的计算图。

#### (2) 推导正向和反向传播方程。

**【解答】**

**1. 增加偏置项后的前向传播方程：**
假设隐藏层偏置为 $\mathbf{b}^{(1)} \in \mathbb{R}^h$，输出层权重仍为 $\mathbf{W}^{(2)}$（为简化，假设输出层不加偏置）：

* $\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$
* $\mathbf{h} = \phi(\mathbf{z})$
* $\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}$
* $L = l(\mathbf{o}, y)$
* $s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2)$
* $J = L + s$

**2. 相应的计算图结构：**
在原书图 4.7.1 的基础上，多出一个方框变量 $\mathbf{b}^{(1)}$。
$\mathbf{b}^{(1)}$ 与 $\mathbf{W}^{(1)}\mathbf{x}$ 的结果通过一个圆形加法算子（$+$）相连，输出为 $\mathbf{z}$。随后的路径保持不变。

**3. 反向传播方程推导：**
我们从上游向下游倒推：

* 对目标函数求导：$\frac{\partial J}{\partial L} = 1$，$ \frac{\partial J}{\partial s} = 1$
* 对输出层变量求导：$\frac{\partial J}{\partial \mathbf{o}} = \frac{\partial L}{\partial \mathbf{o}}$
* 对正则项求导：$\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}$，$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}$
* 对输出层权重求导：$\frac{\partial J}{\partial \mathbf{W}^{(2)}} = \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}$
* 对隐藏层输出求导：$\frac{\partial J}{\partial \mathbf{h}} = (\mathbf{W}^{(2)})^\top \frac{\partial J}{\partial \mathbf{o}}$
* 对仿射变换输出 $\mathbf{z}$ 求导（$\odot$ 表示按元素相乘）：$\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{h}} \odot \phi'(\mathbf{z})$

**【关键：新增加的偏置项偏导】**
根据 $\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$，因为偏置项是直接加在矩阵乘法结果上的，它的全微分系数为 1，所以：


$$\frac{\partial J}{\partial \mathbf{b}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}}$$


*(注：如果考虑小批量数据 Batch，这里需要对整个 Batch 维度的梯度进行求和累加)*。

* 对输入层权重求导：$\frac{\partial J}{\partial \mathbf{W}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}$

---

### 3. 计算本节所描述的模型，用于训练和预测的内存占用。

**【解答】**

* **预测（推理/Inference）阶段：**
* **内存占用非常低。** 预测时只需要计算网络前向传播的最终输出 $\mathbf{o}$。
* 在计算每一层时，**前一层的中间变量可以立即被销毁或覆盖**（例如计算出 $\mathbf{h}$ 之后，$\mathbf{z}$ 的内存就可以直接释放）。
* 因此，预测阶段的最大内存占用仅仅是**网络中单层激活值所需内存的最大值 + 模型权重参数本身的静态显存**。


* **训练（Training）阶段：**
* **内存占用非常高。** 因为反向传播算法（链式法则）在计算靠近输入层的参数梯度（如 $\frac{\partial J}{\partial \mathbf{W}^{(1)}}$）时，需要用到前向传播时产生的中间变量 $\mathbf{z}, \mathbf{h}, \mathbf{x}$ 的值。
* 这意味着，**整个前向传播过程中产生的所有中间激活值，都必须完好无损地保留在显存中**，直到该批次反向传播完全结束。
* 训练阶段的内存占用大约是：**所有隐藏层的激活值总量（与网络深度、Batch Size 成正比） + 模型权重 + 优化器状态（如 Adam 的动量矩阵等）**。通常训练所需的内存是预测的数倍到数十倍。



---

### 4. 假设想计算二阶导数。计算图发生了什么？预计计算需要多长时间？

**【解答】**

1. **计算图发生的变化：**
* 标准的反向传播计算一阶导数，就是构建了一个与前向传播流动方向相反的反向计算图。
* 如果想要计算二阶导数（即对一阶导数再次求导），深度学习框架（如 PyTorch 的 `create_graph=True`）会**把“计算一阶梯度的反向传播过程”也当作新的算子，为其重新构建一层前向计算图**。计算图的体积和节点数量会呈爆发式指数级增长。


2. **预计计算耗时：**
* 时间开销通常是普通训练的 **2 到 3 倍以上**。
* 因为计算二阶导涉及矩阵的 Hessian 矩阵计算或雅可比向量积（Hessian-Vector Product），在计算图上需要做双重的“前向+反向”遍历，且会伴随着极大的显存开销。



---

### 5. 假设计算图对当前拥有的 GPU 来说太大了。

#### (1) 请试着把它划分到多个 GPU 上。

#### (2) 与小批量训练相比，有哪些优点和缺点？

**【解答】**

1. **多 GPU 划分策略（主要有两种）：**
* **模型并行（Model Parallelism）：** 将计算图横向或纵向“切开”。例如，将前几层放在 GPU 0 上计算，将后几层放在 GPU 1 上计算；或者将超大的权重矩阵切分，GPU 0 计算矩阵前半部分，GPU 1 计算后半部分。
* **数据并行（Data Parallelism）：** 如果是因为 Batch Size 太大装不下，可以保持每块 GPU 都有完整的模型，将小批量数据拆分（例如一共 64 个样本，4 块 GPU 每块跑 16 个），最后同步梯度。


2. **与传统单卡小批量训练（Small Batch Size）的优缺点对比（以模型/流水线并行基准而言）：**
* **优点：**
* **突破硬件极限：** 能够训练单块 GPU 根本塞不下的超大规模模型（如千亿参数大模型）。
* **保持大 Batch 的训练稳定性：** 不需要为了塞进显存而被迫把 Batch Size 减小到 1 或 2，从而避免了小批量训练带来的梯度震荡、不收敛以及 Batch Normalization 失效的问题。


* **缺点：**
* **通信开销巨大：** GPU 之间需要频繁通过 PCIe 或 NVLink 传输中间激活值（前向传激活，反向传梯度），如果通信带宽跟不上，GPU 会严重相互等待（产生气泡/Bubble），降低整体计算效率。
* **实现复杂度高：** 代码编写和集群调优非常复杂，远没有直接减小 Batch Size 来得简单直接。