```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据采集
# 从本地文件中读取数据
data = pd.read_csv('user_behavior_data.csv')
print("数据采集完成，已加载到DataFrame中")
# 打印数据的前5条记录
print(data.head())
# 2. 数据清洗与预处理
# 处理缺失值
data = data.dropna()
# 数据类型转换
data['Age'] = data['Age'].astype(int)  # 把年龄转成整数
data['PurchaseAmount'] = data['PurchaseAmount'].astype(float)  # 购买金额转成小数
data['ReviewScore'] = data['ReviewScore'].astype(int)  # 评分转成整数
# 处理异常值
data = data[(data['Age'].between(18, 70)) &  # 保留18 - 70岁的用户
            (data['PurchaseAmount'] > 0) &  # 购买金额必须大于0
            (data['ReviewScore'].between(1, 5))]  # 评分只能是1 - 5分
# 数据标准化（统一数值范围）
data['PurchaseAmount'] = (data['PurchaseAmount'] - data['PurchaseAmount'].mean()) /
data['PurchaseAmount'].std()  # 购买金额标准化
data['ReviewScore'] = (data['ReviewScore'] - data['ReviewScore'].mean()) / data['ReviewScore'].std()
# 评分标准化
# 保存清洗后的数据
data.to_csv('cleaned_user_behavior_data.csv', index=False)
print("数据清洗完成，已保存为'cleaned_user_behavior_data.csv'")
# 3. 数据统计
# 统计每个购买类别的用户数
purchase_category_counts = data['PurchaseCategory'].value_counts()
print("每个购买类别的用户数:\n", purchase_category_counts)
# 统计不同性别的平均购买金额
gender_purchase_amount_mean = data.groupby('Gender')['PurchaseAmount'].mean()
print("不同性别的平均购买金额:\n", gender_purchase_amount_mean)
# 统计不同年龄段的用户数
bins = [18, 25, 35, 45, 55, 65, 70]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
age_group_counts = data['AgeGroup'].value_counts().sort_index()
print("不同年龄段的用户数:\n", age_group_counts)

```
