制定数据清洗规范（每正确回答 1 个规范点得 1 分，最高得 2 分）
1. 数据收集：明确目标，数据来源确认

2. 初步检查：格式统一, 字段一致性检查，时间与日期数据处理

3. 缺失值处理：识别缺失值，处理策略选择： 删除、填充、标记。

4. 异常值检测与处理：异常值检测：采用统计方法（如 IQR 法、Z - score 法）识别异常值；重复数据检测：识别并删除重复的记录。

5. 数据安全与隐私：对于含有敏感信息的数据，进行脱敏处理，合规性检查。

6. 数据标准化 / 归一化：标准化、归一化：对于数值型特征，采用标准化方法（如 Z - score）将数据转化为均值为 0，标准差为 1 的形式。

  制定数据标注规（每正确回答 1 个规范点得 1 分，最高得 3 分）

7. 准备工作：明确目标、定义类别、工具选择

8. 标注指南编写：编写详细的标注指南，一致性检查规则：

9. 标注流程：培训与测试、样本分配、进度跟踪。建立进度跟踪机制，定期检查标注进度，及时解决遇到的问题。

10. 标注执行：逐条标注、标记不确定性、定期保存：定期保存标注进度，防止数据丢失。

11. 质量控制：双重标注、抽样审核、反馈循环：根据审核结果提供反馈给标注员，必要时重新培训或调整标注指南。

12. 后期处理：合并标注结果、格式转换：将标注结果转换为适合模型训练的格式（如 JSON, CSV 等）；数据清洗

```python
import pandas as pd

# 加载数据集并指定编码
file_path = 'medical_data.csv'
data = pd.read_csv(file_path, encoding='gbk')
# 查看数据类型
print(data.dtypes)
# 查看表结构
print(data.info())
# 显示每一列的空缺值数量
print(data.isnull().sum())
# 规范日期格式
data['就诊日期'] = pd.to_datetime(data['就诊日期'])
data['诊断日期'] = pd.to_datetime(data['诊断日期'])
# 重命名列
data.rename(columns={'病人ID': '患者ID'}, inplace=True)
# 查看修改后的表结构
print(data.head())
from datetime import datetime

# 增加诊断延迟和病程
data['诊断延迟'] = (data['诊断日期'] - data['就诊日期']).dt.days
data['病程'] = (datetime(2024, 9, 1) - data['诊断日期']).dt.days
# 删除不合理的数据
data = data[(data['诊断延迟'] >= 0) & (data['年龄'] > 0) & (data['年龄'] < 120)]
# 查看修改后的数据
print(data.describe())
# 删除重复值并记录删除的行数
initial_rows = data.shape[0]
data.drop_duplicates(inplace=True)
deleted_rows = initial_rows - data.shape[0]
print(f'删除的重复行数:{deleted_rows}')
from sklearn.preprocessing import MinMaxScaler

# 对需要归一化的列进行处理
scaler = MinMaxScaler()
columns_to_normalize = ['年龄', '体重', '身高']
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
# 查看归一化后的数据
print(data.head())
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 统计治疗结果分布
treatment_outcome_distribution = data.groupby('疾病类型')['治疗结果'].value_counts().unstack()
# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 根据你的系统调整字体路径
my_font = fm.FontProperties(fname=font_path)
# 绘制柱状图
treatment_outcome_distribution.plot(kind='bar', stacked=True)
plt.title('不同疾病类型的治疗结果分布', fontproperties=my_font)
plt.xlabel('疾病类型', fontproperties=my_font)
plt.ylabel('治疗结果数量', fontproperties=my_font)
plt.xticks(fontproperties=my_font)  # 设置x轴刻度标签的字体
plt.yticks(fontproperties=my_font)  # 设置y轴刻度标签的字体
plt.legend(prop=my_font)  # 设置图例字体
plt.show()
# 绘制散点图
plt.scatter(data['年龄'], data['疾病严重程度'])
plt.title('年龄和疾病严重程度的关系', fontproperties=my_font)
plt.xlabel('年龄', fontproperties=my_font)
plt.ylabel('疾病严重程度', fontproperties=my_font)
plt.xticks(fontproperties=my_font)  # 设置x轴刻度标签的字体
plt.yticks(fontproperties=my_font)  # 设置y轴刻度标签的字体
plt.legend(prop=my_font)  # 设置图例字体
plt.show()
# 保存数据
output_path = '2.1.4_cleaned_data.csv'
data.to_csv(output_path, index=False)
```
