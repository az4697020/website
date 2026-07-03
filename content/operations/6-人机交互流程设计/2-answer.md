```python
import onnxruntime
import numpy as np
from PIL import Image

# 加载ONNX模型
ort_session = onnxruntime.InferenceSession("mnist.onnx")
# 加载图像
image = Image.open("img_test.png").convert('L')  # 转为灰度图
# 图像预处理
image = image.resize((28, 28))  # 调整大小为MNIST模型的输入尺寸
image_array = np.array(image, dtype=np.float32)  # 转为numpy数组
image_array = np.expand_dims(image_array, axis=0)  # 添加batch维度
image_array = np.expand_dims(image_array, axis=0)  # 添加通道维度
# 使用模型对图片进行识别
ort_inputs = {ort_session.get_inputs()[0].name: image_array}
# 执行预测
ort_outs = ort_session.run(None, ort_inputs)
# 获取预测结果
predicted_class = np.argmax(ort_outs[0])
# 输出预测结果
print(f"Predictedclass:{predicted_class}")
```

人机交互的最优方式(每正确回答 1 个优化方式得 1 分，最高得 4 分)
1. 用户界面设计：设计直观简洁界面，有明显上传按钮和数字识别结果展示区，确保用户体验流畅。
2. 自动初始化：系统启动自动加载模型，无需用户操作，保证即时可用。
3. 用户输入：用户可轻松上传手写数字图片进行识别。
4. 高效响应：系统性能优化，迅速响应用户请求，提供即时识别结果，提升体验。

