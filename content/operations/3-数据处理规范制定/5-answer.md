制定数据清洗规范（每正确回答 1 个规范点得 1 分，最高得 2 分）
1. 数据收集：明确目标，数据来源确认

2. 初步检查：格式统一, 字段一致性检查，时间与日期数据处理

3. 缺失值处理：识别缺失值，处理策略选择： 删除、填充、标记。

4. 异常值检测与处理：异常值检测：采用统计方法（如 IQR 法、Z - score 法）识别异常值；重复数据检测：识别并删除重复的记录。

5. 数据安全与隐私：对于含有敏感信息的数据，进行脱敏处理，合规性检查。

6. 数据标准化 / 归一化：标准化、归一化：对于数值型特征，采用标准化方法（如 Z - score）将数据转化为均值为 0，标准差为 1 的形式。

  制定数据标注规范（每正确回答 1 个规范点得 1 分，最高得 3 分）

7. 准备工作：明确目标、定义类别、工具选择

8. 标注指南编写：编写详细的标注指南，一致性检查规则：

9. 标注流程：培训与测试、样本分配、进度跟踪。建立进度跟踪机制，定期检查标注进度，及时解决遇到的问题。

10. 标注执行：逐条标注、标记不确定性、定期保存：定期保存标注进度，防止数据丢失。

11. 质量控制：双重标注、抽样审核、反馈循环：根据审核结果提供反馈给标注员，必要时重新培训或调整标注指南。

12. 后期处理：合并标注结果、格式转换：将标注结果转换为适合模型训练的格式（如 JSON, CSV 等）；数据清洗。

```python
import pandas as pd

# 加载数据集
file_path = '健康咨询客户数据集.csv'
data = pd.read_csv(file_path)
# 查看表的数据类型和表结构
data_info = data.info()
print(data_info)
# 显示每一列的空缺值数量
missing_values = data.isnull().sum()
print(missing_values)
# 删除含有缺失值的行
data_cleaned = data.dropna()
# 或者，可以对特定列进行填充（这里示例用均值填充）
# data['column_name'].fillna(data['column_name'].mean(), inplace=True)
print(data_cleaned.info())
# 转换'Yourage'列的数据类型为整数类型，并处理异常值
data_cleaned.loc[:, 'Yourage'] = pd.to_numeric(data_cleaned['Yourage'], errors='coerce')
data_cleaned = data_cleaned.dropna(subset=['Yourage'])
data_cleaned = data_cleaned[data_cleaned['Yourage'] >= 0]
data_cleaned.loc[:, 'Yourage'] = data_cleaned['Yourage'].astype(int)
print(data_cleaned['Yourage'].dtype)
# 检查和删除重复值
duplicates_removed = data_cleaned.duplicated().sum()
data_cleaned = data_cleaned.drop_duplicates()
print(f"Removed{duplicates_removed}duplicaterows")
from sklearn.preprocessing import LabelEncoder

# 归一化'Howdoyoudescribeyourcurrentleveloffitness?'列
label_encoder = LabelEncoder()
data_cleaned['How do you describe your current level of fitness ?'] =
label_encoder.fit_transform(data_cleaned['Howdoyoudescribeyourcurrentleveloffitness?'])
print(data_cleaned['Howdoyoudescribeyourcurrentleveloffitness?'].unique())
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 去掉列名中的空格
data.columns = data.columns.str.strip()
# 显示数据集的列名
print(data.columns)
# 删除包含缺失值的行
data_cleaned = data.dropna(subset=['Howoftendoyouexercise?'])
# 统计不同健身频率的分布情况
exercise_frequency_counts = data_cleaned['Howoftendoyouexercise?'].value_counts()
# 绘制饼图
plt.figure(figsize=(10, 6))
exercise_frequency_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('DistributionofExerciseFrequency')
plt.ylabel('')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 填充缺失值
data_filled = data.apply(lambda x: x.fillna(x.mode()[0]))
# 划分数据
train_data, test_data = train_test_split(data_filled, test_size=0.2, random_state=42)
# 保存数据
cleaned_file_path = '2.1.5_cleaned_data.csv'
data_filled.to_csv(cleaned_file_path, index=False)
```
