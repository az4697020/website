```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据采集
# 从本地文件中读取数据
data = pd.read_csv('vehicle_traffic_data.csv')
print("数据采集完成，已加载到DataFrame中")
# 打印数据的前5条记录
print(data.head())
# 2. 数据清洗与预处理
# 处理缺失值
data = data.dropna()
# 数据类型转换
data['Age'] = data['Age'].astype(int)  # 年龄转整数
data['Speed'] = data['Speed'].astype(float)  # 车速转小数
data['TravelDistance'] = data['TravelDistance'].astype(float)  # 行驶距离转小数
data['TravelTime'] = data['TravelTime'].astype(float)  # 行驶时间转小数
# 处理异常值
data = data[(data['Age'].between(18, 70)) &  # 保留18 - 70岁驾驶员
            (data['Speed'].between(0, 200)) &  # 车速0 - 200公里 / 小时
            (data['TravelDistance'].between(1, 1000)) &  # 行驶距离1 - 1000公里
            (data['TravelTime'].between(1, 1440))]  # 行驶时间1 - 1440分钟（24小时）
# 保存清洗后的数据
data.to_csv('cleaned_vehicle_traffic_data.csv', index=False)
print("数据清洗完成，已保存为'cleaned_vehicle_traffic_data.csv'")
# 3. 数据合理性审核
# 审核字段合理性
unreasonable_data = data[~((data['Age'].between(18, 70)) &
                           (data['Speed'].between(0, 200)) &
                           (data['TravelDistance'].between(1, 1000)) &
                           (data['TravelTime'].between(1, 1440)))]
print("不合理的数据:\n", unreasonable_data)
# 4. 数据统计
# 统计每种交通事件的发生次数
traffic_event_counts = data['TrafficEvent'].value_counts()
print("每种交通事件的发生次数:\n", traffic_event_counts)
# 统计不同性别的平均车速、行驶距离和行驶时间
gender_stats = data.groupby('Gender').agg({'Speed': 'mean', 'TravelDistance': 'mean', 'TravelTime':
    'mean'})
print("不同性别的平均车速、行驶距离和行驶时间:\n", gender_stats)
# 统计不同年龄段的驾驶员数
age_bins = [18, 25, 35, 45, 55, 65, 70]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-70']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)
age_group_counts = data['AgeGroup'].value_counts()
print("不同年龄段的驾驶员数:\n", age_group_counts)

```
