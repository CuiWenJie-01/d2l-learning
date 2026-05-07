import pandas as pd

# 读取数据
df = pd.read_csv('data/data_file.csv')
# print(df)
 
 # 处理缺失值
inputs,outputs=df.iloc[:,0:2],df.iloc[:,2]
inputs=inputs.fillna(inputs.mean())
print(inputs)
