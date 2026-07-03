------

随着人们健康意识的增强，越来越多的人开始关注日常运动和健康管理。使用提供的训练数据，补全2.2.3.ipynb代码。选择合适的特征，开发一个预测模型，基于个体性别， 个体对运动的看法和个人健康评价来预测个体年龄。利用测试工具对模型进行测试，并对测试结果进行分析，完成测试报告，并运用工具对错误原因进行纠正。

详细说明如下：

| 变量名 | 描述 | 类型 |
|--------|------|------|
| Timestamp | 记录条目时间 | datetime |
| Your name | 参与者姓名 | string |
| Your gender | 参与者性别 | string |
| Your age | 参与者年龄 | string |
| How important is exercise to you? | 锻炼对你的重要性 | integer |
| How do you describe your current level of fitness? | 描述你目前的健康水平 | string |
| How often do you exercise? | 你多久锻炼一次 | string |
| Barriers to exercise | 阻碍你定期锻炼的障碍（如时间不足、动力不足等） | string |
| Forms of exercise | 目前参与的锻炼形式（如跑步、游泳等） | string |
| Factors affecting fitness | 影响健康的因素（如对快餐的易得性、诱惑等） | string |
| How healthy do you consider yourself? | 你认为自己有多健康 | integer |
| Recommended fitness routine to friends? | 你是否曾推荐朋友遵循健身计划 | string |
| Purchased fitness equipment? | 你是否曾购买过健身设备 | string |
| Motivations to exercise | 激励你锻炼的因素（如为了健康、减肥等） | string |

1. 正确加载数据集，并显示前五行的数据
2. 使用随机森林模型进行模型训练，要求设定自变量和因变量，并根据自变量特征进行模型训练，最终将训练好的模型以文件名2.2.3_model.pkl保存到考生文件夹，结果文件以2.2.3_results.txt保存到考生文件夹。
3. 使用测试工具对模型进行测试，并记录测试结果，命名2.2.3_report.txt，保存到考生文件夹
4. 对测试结果进行详细分析，并编写测试报告，包括模型性能评估、错误分析及改进建议，将答案写到答题卷文件中，答题卷文件命名为“2.2.3.docx”，保存到考生文件夹。
5. 运用工具分析算法中错误案例产生的原因并进行纠正，重新得到模型训练结果，以文件名2.2.3_results_xgb.txt保存到考生文件夹。
6. 将以上代码以及运行结果，以html格式保存并命名为2.2.3.html，保存到考生文件夹，考生文件夹命名为“准考证号+身份证后6位”。
