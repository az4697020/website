###### 制定数据清洗规范（每正确回答 1 个规范点得 1 分，最高得 2 分）

1. 数据收集：明确目标，数据来源确认
2. 初步检查：格式统一, 字段一致性检查，时间与日期数据处理
3. 缺失值处理：识别缺失值，处理策略选择： 删除、填充、标记。
4. 异常值检测与处理：异常值检测：采用统计方法（如 IQR 法、Z - score 法）识别异常值；重复数据检测：识别并删除重复的记录。
5. 数据安全与隐私：对于含有敏感信息的数据，进行脱敏处理，合规性检查。
6. 数据标准化 / 归一化：标准化、归一化：对于数值型特征，采用标准化方法（如 Z - score）将数据转化为均值为 0，标准差为 1 的形式。

###### 制定数据标注规范（每正确回答 1 个规范点得 1 分，最高得 3 分）

1. 准备工作：明确目标、定义类别、工具选择
2. 标注指南编写：编写详细的标注指南，一致性检查规则：
3. 标注流程：培训与测试、样本分配、进度跟踪。建立进度跟踪机制，定期检查标注进度，及时解决遇到的问题。
4. 标注执行：逐条标注、标记不确定性、定期保存：定期保存标注进度，防止数据丢失。
5. 质量控制：双重标注、抽样审核、反馈循环：根据审核结果提供反馈给标注员，必要时重新培训或调整标注指南。
6. 后期处理：合并标注结果、格式转换：将标注结果转换为适合模型训练的格式（如 JSON, CSV 等）；数据清洗。

------

```python
import pandas as pd

# 读取一个Excel文件，并将读取到的数据存储在变量data中
data = pd.read_excel('大学生低碳生活行为的影响因素数据集.xlsx')
# 打印出数据集的前5行
print(data.head())
# 处理数据集中的缺失值
initial_row_count = data.shape[0]
data = data.dropna()
final_row_count = data.shape[0]
print(f'处理后数据行数:{final_row_count}, 删除的行数:{initial_row_count - final_row_count}')
# 处理重复行
duplicate_count = data.duplicated().sum()
data = data.drop_duplicates()
print(f'删除的重复行数:{duplicate_count}')
from sklearn.preprocessing import StandardScaler

numerical_features = ['4.您的月生活费○≦1, 000 元 ○1, 001-2, 000元 ○2, 001-3, 000元○≧3, 001元']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
selected_features = ['1.您的性别○男性 ○女性', '2.您的年级○大一 ○大二 ○大三 ○大四',
                     '3.您的生源地○农村 ○城镇（乡镇） ○地县级城市 ○省会城市及直辖市',
                     '4.您的月生活费○≦1, 000元 ○1, 001-2, 000元 ○2, 001-3, 000元 ○≧3, 001元',
                     '5.您进行过绿色低碳的相关生活方式吗?', '6.您觉得“低碳”，与你的生活关系密切吗？',
                     '7.低碳生活是否会成为未来的主流生活方式？', '8.您是否认为低碳生活会提高您的生活质量？']
X = data[selected_features]
# 创建目标变量
y = data['低碳行为积极性']
from sklearn.model_selection import train_test_split

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 保存处理后的数据
cleaned_data = pd.concat([X, y], axis=1)
cleaned_data.to_csv('2.1.2_cleaned_data.csv', index=False)
```

