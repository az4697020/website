------

```PYTHON
import onnxruntime as ort
import numpy as np
import scipy.special
from PIL import Image


# 预处理图像
def preprocess_image(image, resize_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.resize((resize_size, resize_size), Image.BILINEAR)
    w, h = image.size
    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = image.reshape((1,) + image.shape)
    return image


# 加载模型
session = ort.InferenceSession('flower-detection.onnx')  # ONNXRuntime模型加载接口
# 加载类别标签
with open('labels.txt', 'r') as f:  # 打开标签文件
    labels = [line.strip() for line in f.readlines()]
# 获取模型输入和输出的名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
# 加载图片
image = Image.open('flower_test.png').convert('RGB')  # PIL图像加载方法
# 预处理图片
processed_image = preprocess_image(image)  # 调用预处理函数
# 确保输入数据是 float32 类型
processed_image = processed_image.astype(np.float32)
# 进行图片识别
output = session.run([output_name], {input_name: processed_image})[0]  # 执行模型推理
# 应用 softmax 函数获取识别分类后的准确率
accuracy = scipy.special.softmax(output, axis=-1)  # 概率归一化
# 获取预测的类别索引
predicted_idx = np.argmax(accuracy)
# 获取预测的准确值（转换为百分比）
prob_percentage = accuracy[0, predicted_idx] * 100
# 获取预测的类别标签
predicted_label = labels[predicted_idx]
# 输出预测结果，包含百分比形式的概率
print(f"Predictedclass:{predicted_label}, Accuracy:{prob_percentage:.2f}%")
```

人机交互的最优方式(每正确回答 1 个优化方式得 1 分，最高得 5 分)
1. 构建用户友好的界面

  设计一个直观、用户友好的界面，具有明显的图片上传按钮和专门的识别结果展示区域， 简化了用户操作，提升了体验。

2. 系统初始化时的预加载

  系统启动或用户访问时，会预加载花朵识别模型和类别标签，快速响应请求，提供即时分析服务。

3. 便捷的图片提交方式

  用户可通过上传按钮轻松提交待识别的花朵图片，快速开始识别体验。 

4. 自动化图像调整

  图片上传后，系统自动进行尺寸调整、裁剪和归一化等预处理，确保符合模型输入要求， 提高识别准确性。

5. 智能识别技术的应用

  预处理后的图片送入模型分析，系统根据模型成果识别花朵类型。
