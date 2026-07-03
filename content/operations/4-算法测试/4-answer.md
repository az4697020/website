```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from xgboost import XGBRegressor

# 加载数据集
file_path = '大学生低碳生活行为的影响因素数据集.xlsx'
data = pd.read_excel(file_path)
# 显示数据集的前五行
print(data.head())
# 删除不必要的列并处理分类变量
data_cleaned = data.drop(columns=['序号', '所用时间'])  # 删除不必要的列
data_cleaned = pd.get_dummies(data_cleaned, drop_first=True)  # 将分类变量转换为哑变量 / 指示变量
# 定义目标变量和特征
target = '5.您进行过绿色低碳的相关生活方式吗?'  # 确保这是目标变量
features = data_cleaned.drop(columns=[target])
# 定义自变量因变量
X = features
y = data_cleaned[target]
# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
# 保存训练好的模型
model_filename = '2.2.4_model.pkl'
joblib.dump(model, model_filename)
# 进行预测
y_pred = model.predict(X_test)
# 将结果保存到文本文件中
results = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
results_filename = '2.2.4_results.txt'
results.to_csv(results_filename, index=False, sep='\t')  # 使用制表符分隔值保存到文本文件
# 将测试结果保存到报告文件中
report_filename = '2.2.4_report.txt'
with open(report_filename, 'w') as f:
    f.write(f'均方误差:{mean_squared_error(y_test, y_pred)}\n')
f.write(f'决定系数:{r2_score(y_test, y_pred)}\n')
# 分析并纠正错误（示例：使用XGBoost）
# 训练XGBoost模型
xgb_model = XGBRegressor(n_estimators=1000,  # 增加树的数量
                         learning_rate=0.05,  # 降低学习率
                         max_depth=5,  # 调整树的深度
                         subsample=0.8,  # 调整样本采样比例
                         colsample_bytree=0.8)  # 调整特征采样比例
xgb_model.fit(X_train, y_train)
# 使用XGBoost模型进行预测
y_pred_xg = xgb_model.predict(X_test)
# 将XGBoost结果保存到文本文件中
results_xg_filename = '2.2.4_results_xg.txt'
results_xg = pd.DataFrame({'实际值': y_test, '预测值': y_pred_xg})
results_xg.to_csv(results_xg_filename, index=False, sep='\t')  # 使用制表符分隔值保存到文本文件
# 将XGBoost测试结果保存到报告文件中
report_filename_xgb = '2.2.4_report_xgb.txt'
with open(report_filename_xgb, 'w') as f:
    f.write(f'均方误差:{mean_squared_error(y_test, y_pred_xg)}\n')
f.write(f'决定系数:{r2_score(y_test, y_pred_xg)}\n')
```

针对线性回归模型

1. 模型性能

   | 均方误差（MSE） | 0.02467956371342613 |
   | --------------- | ------------------- |
   | 决定系数（R²）  | 0.18477828249844008 |

2. 错误分析

   1. 数据处理层面

      缺失值处理缺失：数据集中可能存在缺失值，但代码未进行任何处理。缺失值可能会影响模型训练的准确性，导致模型在学习过程中出现偏差，进而影响预测结果。例如，若某些样本在关键特征上存在缺失值，模型可能无法准确捕捉这些样本的特征信息，使得训练出的模型在泛化时表现不佳。特征工程不足：仅删除了‘序号’和‘所用时间’列，未对其他特征进行深入分析和处理。可能存在特征之间的相关性未被充分挖掘，或者某些特征对目标变量的影响并不显著，却被保留在数据集中，增加了模型训练的复杂性和噪声。例如，一些特征可能存在共线性，这会干扰模型的学习和预测能力。编码方式单一：使用pd.get_dummies 将分类变量转换为哑变量，这种方式可能会导致数据稀疏，增加模型训练的计算量。对于一些具有内在顺序关系的分类变量，如 “您的年级”，直接进行独热编码可能无法充分利用其顺序信息，影响模型的性能。

   2. 结果分析与应用层面

      缺乏对错误案例的分析：没有对预测错误的样本进行深入分析，不清楚是哪些样本导致了模型误差较大，以及错误产生的原因。这使得难以针对性地改进模型，无法有效提高模型的预测准确性。例如，通过分析错误案例，可以发现某些特定特征组合下模型的预测效果较差，从而对这些特征进行进一步处理或调整模型。模型可解释性未充分考虑：在实际应用中，模型的可解释性很重要。无论是线性回归还是 XGBoost 模型，都没有对模型的预测结果进行解释，难以理解模型是如何做出决策的。这在一些需要对结果进行解释的场景中，如政策制定、用户行为分析等，会限制模型的应用价值。

   3. 改进建议

      1. 数据处理优化

         处理缺失值：使用合适的方法填充缺失值，如均值填充、中位数填充或 K 近邻算法填充。例如，对于 “月生活费”“生源地” 等特征的缺失值，可以根据其他样本的情况进行合理填充。若某一特征缺失值较多且对模型影响不大，可考虑删除该特征。

         特征选择与工程：运用特征选择技术，如相关性分析、方差分析等，筛选出与目标变量相关性较高的特征，去除冗余和噪声特征 。计算各特征与目标变量 “5. 您进行过绿色低碳的相关生活方式吗？” 的相关性，保留相关性高的特征，减少模型训练的复杂性。还可以尝试创建新的特征，如对相关特征进行组合、计算特征之间的比例等，以提升模型的预测能力。比如将 “您从哪些途径获取低碳信息” 相关的多个特征进行组合，形成一个新的综合特征。改进编码方式：对于有序分类变量，如 “您的年级”，采用更合适的编码方法，如标签编码（LabelEncoding）或序数编码（OrdinalEncoding），保留其顺序信息。对于其他分类变量，在使用独热编码时，可以结合特征重要性分析，减少不必要的编码维度，降低数据稀疏性。

      2. 结果分析与应用拓展

         深入分析错误案例：对比预测错误的样本和正确预测的样本，分析它们在特征上的差异，找出导致错误的关键因素。例如，可以绘制不同特征下的预测误差分布，观察哪些特征对预测结果影响较大。对于预测错误较多的样本，可以进一步分析其特征组合，针对性地调整模型或数据处理方式。

         增强模型可解释性：对于线性回归模型，可以分析回归系数，了解各个特征对目标变量的影响方向和程度。对于 XGBoost 模型，可以使用特征重要性分析，查看哪些特征在模型决策中起到关键作用。通过这些方法，使模型的预测结果更具可解释性，便于在实际应用中理解和应用。

         模型应用拓展：将模型应用到实际场景中，如预测不同群体的低碳生活行为，为相关政策制定提供参考。根据模型的预测结果，分析不同因素对低碳生活行为的影响，提出针对性的建议，如加强对特定群体的低碳宣传教育、完善相关设施等。
