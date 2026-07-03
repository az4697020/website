###### 数据清洗规范（答对2点即可）

1. 数据加载：使用 pandas 库加载数据集，检查数据的基本结构和类型。
2. 检查缺失值：统计每列的缺失值数量，并删除包含缺失值的行以确保数据完整性。
3. 转换与处理异常值: 将数值列（如“horsepower”）转换为数值类型，并处理无法转换的值（例如，将其变为缺失值）。
4. 数据标准化: 对数值型数据进行标准化，以消除量纲影响，使用标准化方法。
5. 保存清洗后的数据: 将经过清洗和处理后的数据保存为新的 CSV 文件，以便后续使用。

###### 数据标注规范（答对3点即可）

1. 数据来源：标注数据的来源，包括数据集的名称、获取日期和数据提供者。
2. 数据描述：提供详细的数据描述，包括每列数据的含义、单位和可能的取值范围。
3. 特征选择: 确定对目标变量预测最有用的特征。
4. 目标变量设定: 将数据集中用于预测的目标变量定义为“mpg”（燃油效率）。
5. 数据划分: 将数据分为训练集和测试集，通常采用 80 / 20 的比例，以便于模型的训练和评估。
6. 保存处理后的数据：保存处理后的数据，并记录保存文件的路径和文件名。
7. 数据清洗和标注规范文档

```python
import pandas as pd

# 加载数据集并显示数据集的前五行
file_path = 'auto-mpg.csv'
data = pd.read_csv(file_path)
print("数据集的前五行:")
print(data.head())
# 显示每一列的数据类型
print(data.dtypes)
# 检查缺失值并删除缺失值所在的行
print("\n检查缺失值:")
print(data.isnull().sum())
data = data.dropna()
# 将'horsepower'列转换为数值类型，并处理转换中的异常值
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data = data.dropna(subset=['horsepower'])
# 显示每一列的数据类型
print(data.horsepower.dtypes)
# 检查清洗后的缺失值
print("\n检查清洗后的缺失值:")
print(data.isnull().sum())
from sklearn.preprocessing import StandardScaler

# 对数值型数据进行标准化处理
numerical_features = ['displacement', 'horsepower', 'weight', 'acceleration']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
from sklearn.model_selection import train_test_split

# 选择特征和目标变量
selected_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
X = data[selected_features]
y = data['mpg']
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 将特征和目标变量合并到一个数据框中
cleaned_data = X.copy()
cleaned_data['mpg'] = y
# 保存清洗和处理后的数据
cleaned_data.to_csv('2.1.1_cleaned_data.csv', index=False)
# 打印消息指示文件已保存
print("\n清洗后的数据已保存到 2.1.1_cleaned_data.csv")
```

