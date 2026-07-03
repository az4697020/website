```python
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

# 模型加载
session = ort.InferenceSession('resnet.onnx')  # 补全onnxruntime接口
# 加载类别标签
labels_path = 'labels.txt'
with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]
# 获取模型输入和输出的名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
# 加载图片
image = Image.open('img_test.jpg').convert('RGB')  # 调用PIL的图片加载方法
# 预处理图片
processed_image = preprocess_image(image)  # 调用预定义的预处理函数
# 确保输入数据是 float32 类型
processed_image = processed_image.astype(np.float32)
# 进行图片识别
output = session.run([output_name], {input_name: processed_image})[0]  # 执行模型推理
# 应用 softmax 函数获取概率
probabilities = scipy.special.softmax(output, axis=-1)  # 调用scipy的softmax函数
# 获取最高的5个概率和对应的类别索引
top5_idx = np.argsort(probabilities[0])[- 5:][:: - 1]  # 使用numpy的排序函数
top5_prob = probabilities[0][top5_idx]
# 打印结果
print("Top5predictedclasses:")
for i in range(5):
    print(f"{i + 1}:{labels[top5_idx[i]]}-Probability:{top5_prob[i]}")
```

人机交互的最优方式(每正确回答 1 个优化方式得 1 分，最高得 2 分)
1. 用户界面设计直观易用：创建简洁的用户界面，方便用户上传图片和查看识别结果，无需复杂操作。
2. 模型即时可用：系统启动时预加载模型和标签，用户可立即进行图片识别，减少等待时间。
3. 图像智能分析：利用高级算法深度分析预处理图像，提供精确识别服务。
4. 高效系统性能：优化响应速度和模型处理效率，即使高并发也能快速准确服务。
