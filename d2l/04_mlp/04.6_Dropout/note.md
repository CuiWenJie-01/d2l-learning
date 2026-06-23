### 参数m的含义作用
          
在这段代码中，参数 `m` 代表 **神经网络中的每个子模块（submodule）**。

当调用 `net.apply(init_weights)` 时，PyTorch 会**递归遍历** `net` 中的所有子模块（如 `nn.Flatten`、`nn.Linear`、`nn.ReLU`、`nn.Dropout` 等），并将每个子模块作为参数 `m` 传入 `init_weights` 函数。

函数内部通过 `type(m) == nn.Linear` 判断当前模块是否是线性层，如果是，则初始化其权重（使用标准差为 0.01 的正态分布）和偏置（初始化为 0）。

这是 PyTorch 中一种**批量初始化网络参数**的标准模式。