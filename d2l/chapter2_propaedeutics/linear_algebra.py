import torch

A=torch.arange(12,dtype=torch.float32).reshape(3,4)
X=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
# print(A.sum(axis=1)) # 列求和
'''
keepdims=True 求和后保留维度，方便后续广播运算。
'''
# print(A/A.sum(axis=1,keepdims=True)) # 列归一化
print(X)
'''
计算张量的范数:
所有元素平方
全部相加
最后开方

会把整个张量看成一个超长向量
'''
print(torch.linalg.norm(X))