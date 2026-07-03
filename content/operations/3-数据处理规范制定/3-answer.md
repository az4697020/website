###### 制定数据清洗规范（每正确回答 1 个规范点得 1 分，最高得 3 分）

1. 数据收集：明确目标，数据来源确认
2. 初步检查：格式统一, 字段一致性检查，时间与日期数据处理
3. 缺失值处理：识别缺失值，处理策略选择： 删除、填充、标记。
4. 异常值检测与处理：异常值检测：采用统计方法（如 IQR 法、Z - score 法）识别异常值；重复数据检测：识别并删除重复的记录。
5. 数据安全与隐私：对于含有敏感信息的数据，进行脱敏处理，合规性检查。
6. 数据标准化 / 归一化：标准化、归一化：对于数值型特征，采用标准化方法（如 Z - score）将数据转化为均值为 0，标准差为 1 的形式。

###### 制定特征工程规范（每正确回答 1 个规范点得 1 分，最高得 2 分）

1. 数据理解与准备:明确业务目标：理解项目的具体业务需求和目标，这将指导特征的选择和构建
2. 特征提取：根据领域知识，从原始数据中提取有意义的新特征。例如，在金融领域可以计算贷款利率的变化率作为新特征。
3. 特征变换： 数学变换、分箱（Binning）、交互特征。
4. 特征选择：过滤方法、包装方法、嵌入方法、降维技术。
5. 特征验证与评估：交叉验证、模型性能评估、特征重要性分析。

------

```python
import pandas as pd

# Loadthedata
file_path = 'finance数据集.csv'
data = pd.read_csv(file_path)
# 显示前五行的数据
data.head()
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图像尺寸
plt.figure(figsize=(12, 8))
# 识别数值列用于箱线图
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# 创建箱线图
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
sns.boxplot(x=data[col])
plt.title(col)
plt.tight_layout()
plt.show()
# 使用IQR处理异常值
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
# 移除异常值
data_cleaned = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 *
                                                                                       IQR))).any(axis=1)]
# 检查重复值
duplicates = data_cleaned.duplicated()
num_duplicates = duplicates.sum()
data_cleaned = data_cleaned[~duplicates]
print(f'删除的重复行数:{num_duplicates}')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_cleaned[numeric_cols] = scaler.fit_transform(data_cleaned[numeric_cols])
# 将SeriousDlqin2yrs设为目标变量
target_variable = 'SeriousDlqin2yrs'
from sklearn.model_selection import train_test_split

# 定义特征和目标
X = data_cleaned.drop(columns=[target_variable])
y = data_cleaned[target_variable]
# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 显示划分后的数据形状
print(f'训练数据形状:{X_train.shape}')
print(f'测试数据形状:{X_test.shape}')
# 保存清洗后的数据到CSV
cleaned_file_path = '2.1.3_cleaned_data.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)
```

