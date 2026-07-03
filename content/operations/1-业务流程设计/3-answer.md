```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('credit_data.csv')
# 1. 数据完整性审核
missing_values = data.isnull().sum()  # 统计每列有多少空白格子
duplicate_values = data.duplicated().sum()  # 统计有多少行完全重复
print("缺失值统计:")
print(missing_values)
print("重复值统计:")
print(duplicate_values)
# 2. 数据合理性审核
data['is_age_valid'] = data['Age'].between(18, 70)  # 检查年龄是否在18 - 70岁之间
data['is_income_valid'] = data['Income'] > 2000  # 检查收入是否高于2000元
data['is_loan_amount_valid'] = data['LoanAmount'] < (data['Income'] * 5)  # 检查贷款是否超过收入的5倍
data['is_credit_score_valid'] = data['CreditScore'].between(300, 850)  # 检查信用分是否在300 - 850分
# 综合检查：所有条件都满足才算有效数据
validity_checks = data[['is_age_valid', 'is_income_valid', 'is_loan_amount_valid', 'is_credit_score_valid']].all(axis=1)
data['is_valid'] = validity_checks
print("数据合理性检查:")
print(data[['is_age_valid', 'is_income_valid', 'is_loan_amount_valid', 'is_credit_score_valid', 'is_valid']].describe())
# 3. 数据清洗和异常值处理
cleaned_data = data[data['is_valid']]  # 只保留有效数据
cleaned_data = cleaned_data.drop(
    columns=['is_age_valid', 'is_income_valid', 'is_loan_amount_valid', 'is_credit_score_valid', 'is_valid'])
cleaned_data.to_csv('cleaned_credit_data.csv', index=False)  # 保存干净数据
print("数据清洗完成，已保存为'cleaned_credit_data.csv'")
```
