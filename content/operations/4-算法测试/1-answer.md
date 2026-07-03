```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 加载数据
file_path = 'finance数据集.csv'
data = pd.read_csv(file_path)
# 显示前五行的数据
print(data.head())
# 选择自变量和因变量
X = data.drop(['SeriousDlqin2yrs', 'Unnamed:0'], axis=1)
y = data['SeriousDlqin2yrs']
# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练Logistic回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# 保存模型
with open('2.2.1_model.pkl', 'wb') as file:
    pickle.dump(model, file)
# 预测并保存结果
y_pred = model.predict(X_test)
pd.DataFrame(y_pred, columns=['预测结果']).to_csv('2.2.1_results.txt', index=False)
# 生成测试报告
report = classification_report(y_test, y_pred, zero_division=1)
with open('2.2.1_report.txt', 'w') as file:
    file.write(report)
# 分析测试结果
accuracy = (y_test == y_pred).mean()
print(f"模型准确率:{accuracy:.2f}")
# 处理数据不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# 重新训练模型
model.fit(X_resampled, y_resampled)
# 重新预测
y_pred_resampled = model.predict(X_test)
# 保存新结果
pd.DataFrame(y_pred_resampled, columns=['预测结果']).to_csv('2.2.1_results_xg.txt', index=False)
# 生成新的测试报告
report_resampled = classification_report(y_test, y_pred_resampled, zero_division=1)
with open('2.2.1_report_xg.txt', 'w') as file:
    file.write(report_resampled)
# 分析新的测试结果
accuracy_resampled = (y_test == y_pred_resampled).mean()
print(f"重新采样后的模型准确率:{accuracy_resampled:.2f}")
```

主观题：

1. 填写两个 report 中的数据

2. 错误分析：

   1. 数据不平衡的影响

      - 未使用 SMOTE 时，0 怎么样，1 怎么样，0（未违约的）占主导地位，1（违约的）比较少，模型很难识别

      - 使用 SOMTE 后，0 的召回率从 XX 到 XX 了，表明模型回归得到了明显改善

   1. 类别 0 和类别 1 之间的权衡

      - 未使用 SMOTE 时，模型高度偏向 0，各方面指标都很好

      - 使用 SMOTE 后，使用后正类的召回率提高了，

1. 改进建议
   1. 继续优化数据处理的方式
   2. 继续改进模型（或者根据实际情况选择适当的模型）
   3. 根据实际情况进行参数调优
   4. 改进数据特征工程的选择

针对 Logistic 模型

1. 模型性能

|                   | precision | recall | f1-score | support |
| ----------------- | --------- | ------ | -------- | ------- |
| 0（没有严重逾期） | 0.95      | 0.99   | 0.97     | 26779   |
| 1（有严重逾期）   | 0.57      | 0.14   | 0.22     | 1737    |

1. 错误分析

   - 0（没有严重逾期）：

     准确率很高，召回率也很高，表明模型在这一类别上的性能非常好。

     可能的错误主要来自于少数漏报情况，即极少数实际没有严重逾期的样本被错误预测为有严重逾期。

   - 1（有严重逾期）：

     准确率较低，召回率也很低，F1 - Score 仅为 0.22，表明模型在这一类别上的性能较差。主要问题在于大量的漏报（真正有严重逾期的样本被预测为没有）和一定的误报（将没有严重逾期的样本预测为有）。

1. 改进建议

   1. 数据处理策略调整

      重采样技术：由于数据集存在明显的不平衡，可以考虑使用过采样（如 SMOTE）或欠采样技术来平衡两个类别的数量。

   2. 特征工程优化

      特征选择：仔细审查现有特征，去除冗余或不相关的特征，可能有助于提升模型性能。

      特征构造：尝试创建新的、更具区分力的特征，如基于现有特征的交互项或衍生指标。
   
