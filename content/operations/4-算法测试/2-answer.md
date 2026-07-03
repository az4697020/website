```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import RandomForestRegressor

# 加载数据集
df = pd.read_csv('auto-mpg.csv')
# 显示前五行数据
print(df.head())
# 处理缺失值
# 将'horsepower'列中的所有值转换为数值类型
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
# 删除包含缺失值的行
df = df.dropna()
# 选择相关特征进行建模
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'modelyear', 'origin']]
y = df['mpg']
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建包含标准化和线性回归的管道
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('linreg', LinearRegression())])
# 训练模型
pipeline.fit(X_train, y_train)
# 保存训练好的模型
with open('2.2.2_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
# 预测并保存结果
y_pred = pipeline.predict(X_test)
results_df = pd.DataFrame(y_pred, columns=['预测结果'])
results_df.to_csv('2.2.2_results.txt', index=False)
# 测试模型
with open('2.2.2_report.txt', 'w') as results_file:
    results_file.write(f'训练集得分:{pipeline.score(X_train, y_train)}\n')
results_file.write(f'测试集得分:{pipeline.score(X_test, y_test)}\n')
# 训练一个随机森林回归模型作为替代模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# 使用随机森林模型进行预测
y_pred_rf = rf_model.predict(X_test)
# 保存新的结果
results_rf_df = pd.DataFrame(y_pred_rf, columns=['预测结果'])
results_rf_df.to_csv('2.2.2_results_rf.txt', index=False)
# 测试模型并保存得分
with open('2.2.2_report_rf.txt', 'w') as results_rf_file:
    results_rf_file.write(f'训练集得分:{rf_model.score(X_train, y_train)}\n')
results_rf_file.write(f'测试集得分:{rf_model.score(X_test, y_test)}\n')
```

针对线性回归模型

1. 模型性能

   | 训练集得分 | 0.9810141307577177 |
   | ---------- | ------------------ |
   | 测试集得分 | 0.8881482175005623 |

2. 错误分析

   1. 模型性能方面

      线性回归模型的训练集得分 0.826001578671067 和测试集得分0.7901500386760347，随机森林回归模型的训练集得分 0.9810141307577177 和测试集得分 0.8881482175005623。虽然随机森林模型在训练集和测试集上的表现均优于线性回归模型，但两个模型的测试集得分与训练集得分存在一定差距，表明可能存在过拟合或欠拟合问题。线性回归模型得分相对较低，说明其对数据的拟合能力可能不足，无法很好地捕捉数据中的复杂关系；随机森林模型训练集得分过高，可能存在过拟合，在训练过程中过度学习了训练数据的细节，导致在测试集上的泛化能力受限。

   2. 数据处理方面

      数据集中存在缺失值，如 “horsepower” 列有部分数据缺失。在数据处理时仅将“horsepower” 列中的所有值转换为数值类型并删除包含缺失值的行，这种处理方式可能会丢失大量数据，影响模型的训练效果。此外，对于 “horsepower” 缺失值较多的情况，简单删除可能导致数据信息损失严重，应考虑更合理的填补方法，如均值、中位数填补，或基于其他特征进行预测填补。

   3. 改进建议

      1. 模型性能优化

         尝试使用其他回归模型进行对比实验，如决策树回归、支持向量机回归等，选择性能最优的模型作为最终的预测模型。可以使用交叉验证的方法来评估不同模型的性能，确保模型的稳定性和泛化能力。

         对现有模型进行调优。对于线性回归模型，可以尝试对数据进行特征工程，如特征缩放、特征组合等，以提高模型的拟合能力；对于随机森林模型，可以使用网格搜索或随机搜索等方法对超参数进行调优，寻找最优的超参数组合，降低模型的过拟合风险，提高模型在测试集上的性能。

      2. 数据处理改进

         对于缺失值处理，除了删除缺失值行外，可以尝试使用更复杂的填补方法。如对于“horsepower” 列的缺失值，可以根据发动机气缸数量、排量等相关特征，使用 K 近邻算法（KNN）或其他回归方法进行预测填补，在保留数据完整性的同时，减少缺失值对模型训练的影响。

         对数据进行特征选择，去除对燃油效率影响较小或相关性较低的特征，降低数据维度，提高模型训练效率和性能。可以使用相关性分析、方差分析等方法来选择重要特征。
