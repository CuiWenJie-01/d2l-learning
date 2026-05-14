import pandas as pd

# 读取数据
df = pd.read_csv('data/data_file.csv')
# print(df)
 
# 处理缺失值(使用插值法)
inputs,outputs=df.iloc[:,0:2],df.iloc[:,2]
inputs=inputs.fillna(inputs.mean(numeric_only=True))

inputs=pd.get_dummies(inputs,dummy_na=True)
# print(inputs)

# 转换为张量格式
import torch
X=torch.tensor(inputs.to_numpy(dtype=float)).cuda()
y=torch.tensor(outputs.to_numpy(dtype=float)).cuda()
print(X)
print(y)
