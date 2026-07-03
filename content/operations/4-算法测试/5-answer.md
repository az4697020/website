```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据集
df = pd.read_csv('fitnessanalysis.csv')
# 显示前五行数据
print(df.head())
# 选择相关特征进行建模
X = df[['Your gender ', 'How import ant is exercise to you ?', 'How healthy do you consider yourself?']]
X = pd.get_dummies(X)  # 将分类变量转为数值变量
# 设为目标变量
y = df['daily_steps']  # 假设目标变量为年龄字段
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建并训练决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
# 保存训练好的模型
with open('2.2.5_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)
# 进行预测
y_pred = dt_model.predict(X_test)
# 将结果保存到文本文件中
results = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
results_filename = '2.2.5_results.txt'
results.to_csv(results_filename, index=False, sep='\t')
# 将测试结果保存到报告文件中
report_filename = '2.2.5_report.txt'
with open(report_filename, 'w') as f:
    f.write(f'均方误差:{mean_squared_error(y_test, y_pred)}\n')
    f.write(f'平均绝对误差:{mean_absolute_error(y_test, y_pred)}\n')
    f.write(f'决定系数:{r2_score(y_test, y_pred)}\n')
```

1. 模型性能

| 均方误差（MSE）   | 8096170.758224316   |
| ----------------- | ------------------- |
| 平均绝对误差(MAE) | 2421.827880665033   |
| 决定系数（R²）    | -0.1541458336017123 |

2. 错误分析

  1. 数据处理方面

    缺失值处理缺失：代码未对数据集中可能存在的缺失值进行处理。缺失值可能会影响模型训练的准确性，导致模型在学习过程中产生偏差，进而影响预测结果。例如，若 “如何评价自己目前的健康状况” 这一特征存在缺失值，模型在训练时可能无法准确捕捉这些样本的特征信息，使得训练出的模型在泛化时表现不佳。

    特征选择较简单：仅选择了‘Your gender ’‘How import ant is exercise to you?’‘Howhealthy do you consider yourself?’这三个特征进行建模，可能遗漏了其他对目标变量“daily_steps” 有重要影响的特征。如 “锻炼频率”“锻炼形式”“饮食是否健康” 等特征，可能与每日步数存在较强的相关性，忽略这些特征会导致模型无法充分学习数据中的信息，降低模型的预测能力。

    编码方式问题：对分类变量使用 pd.get_dummies 进行编码，这种方式会增加特征的维度，可能导致数据稀疏，增加模型训练的计算量和过拟合的风险。对于一些具有内在顺序关系的分类变量，如 “如何评价自己目前的健康状况”，直接进行独热编码可能无法充分利用其顺序信息，影响模型的性能。

  2. 模型评估方面

     评估指标单一：仅使用均方误差（MSE）、平均绝对误差（MAE）和决定系数（\(R^2\)）来评估模型性能，过于片面。这三个指标虽然能反映模型的部分性能，但无法全面评估模型在不同数据分布和场景下的表现。例如，对于预测步数这样的实际问题，还可以考虑其他指标，如平均绝对百分比误差（MAPE），它能更好地反映预测值与实际值之间的相对误差，对于评估模型的预测精度更有意义。

     未进行交叉验证：在模型训练和评估过程中，没有进行交叉验证。交叉验证可以通过将数据集多次划分成训练集和测试集，对模型进行多次训练和评估，从而更准确地评估模型的泛化能力。未进行交叉验证可能导致模型评估结果的偏差，无法真实反映模型在未知数据上的性能。

  3. 改进建议

     1. 数据处理

        处理缺失值：原始代码未对数据集中的缺失值进行处理。可使用均值、中位数填充，或采用更复杂的插补方法，如 K 近邻插补，确保数据的完整性，避免缺失值对模型训练和预测的影响。

        特征工程优化：仅选择了三个特征进行建模，可能无法充分捕捉数据中的信息。可考虑增加更多相关特征，如年龄、运动频率、饮食均衡情况等；对现有特征进行组合、变换，生成新的特征，如计算运动重要性与健康自评的乘积，以挖掘特征间的潜在关系。异常值处理：数据中可能存在异常值，如极端的每日步数。可通过可视化（如箱线图）或统计方法识别异常值，然后采用缩尾法、盖帽法等进行处理，防止异常值对模型性能的干扰。数据标准化 / 归一化：对数值型特征进行标准化（如 Z - score 标准化）或归一化（如Min - Max 归一化）处理，可提升模型的训练速度和稳定性，尤其对于一些对尺度敏感的算法，能使其更好地收敛。

     2. 模型评估

        交叉验证：目前仅进行了一次训练集和测试集的划分，结果可能具有随机性。建议使用交叉验证，如 K 折交叉验证，多次划分数据集进行训练和评估，取平均结果，从而更可靠地评估模型性能。使用更多评估指标：除了均方误差（MSE）、平均绝对误差（MAE）和决定系数（\(R^2\)），还可引入其他指标，如均方根误差（RMSE），它能更直观地反映预测值与真实值之间的平均误差程度；平均绝对百分比误差（MAPE），可衡量预测值的相对误差，帮助了解模型在不同量级数据上的表现。

        模型比较：决策树回归模型不一定是最优选择。可尝试其他回归模型，如线性回归、随机森林回归、支持向量回归等，对比不同模型的评估指标，选择性能最佳的模型。绘制学习曲线：绘制学习曲线观察模型在训练集和验证集上的误差随训练样本数量的变化情况，判断模型是否存在过拟合或欠拟合现象，以便及时调整模型复杂度。
