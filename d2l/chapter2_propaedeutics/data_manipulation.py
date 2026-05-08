import torch # 导入torch库
# 生成一个12维的张量
# X=torch.arange(12)
# print(x)
#print(x.shape)
# print(x.numel())
# x=x.reshape(3,4)
# print(x)
# print(torch.zeros((2,3,4)))
# print(torch.ones((2,3,4)))
X=torch.zeros((2,3,4))
# print(X.shape)
# print(len(X)) #默认是第0个维度的长度
# print(X.shape[0]) # 第0个维度的长度

A=torch.arange(12,dtype=torch.float32).reshape(3,4)
print(A.sum(axis=1)) # 列求和
# print(A/A.sum(axis=1,keepdims=True)) # 列归一化