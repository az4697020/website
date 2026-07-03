```python
# 导入必要的库
import onnx
import numpy as np
from PIL import Image
import onnxruntime as ort


# 定义预处理函数，用于将图片转换为模型所需的输入格式
def preprocess(image_path):
    input_shape = (1, 1, 64, 64)  # 模型输入期望的形状，这里是 (N, C, H, W)，N = batchsize, C=channels, H=height, W=width
    img = Image.open(image_path).convert('L')  # 打开图像文件并将其转换为灰度图
    img = img.resize((64, 64), Image.ANTIALIAS)  # 调整图像大小到模型输入所需的尺寸
    img_data = np.array(img, dtype=np.float32)  # 将PIL图像对象转换为numpy数组，并确保数据类型是float32
    # 调整数组的形状以匹配模型输入的形状
    img_data = np.expand_dims(img_data, axis=0)  # 添加 batch 维度
    img_data = np.expand_dims(img_data, axis=1)  # 添加 channel 维度
    assert img_data.shape == input_shape, f"Expectedshape{input_shape}, but got {img_data.shape}"  # 确保最终的形状与模型输入要求的形状一致
    return img_data  # 返回预处理后的图像数据


# 定义情感类别与数字标签的映射表
emotion_table = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6,
                 'contempt': 7}
# 加载模型
ort_session = ort.InferenceSession('emotion-ferplus.onnx')  # 使用onnxruntime创建一个会话，用于加载并运行模型
# 加载本地图片并进行预处理
input_data = preprocess('img_test.png')
# 准备输入数据，确保其符合模型输入的要求
ort_inputs = {ort_session.get_inputs()[0].name: input_data}  # ort_session.get_inputs()[0].name是获取模型的第一个输入的名字
# 运行模型，进行预测
ort_outs = ort_session.run(None, ort_inputs)
# 解码模型输出，找到预测概率最高的情感类别
predicted_label = np.argmax(ort_outs[0])
# 根据预测的标签找到对应的情感名称
predicted_emotion = list(emotion_table.keys())[predicted_label]
# 输出预测的情感
print(f"Predictedemotion:{predicted_emotion}")
```

人机交互的最优方式(每正确回答 1 个优化方式得 1 分，最高得 6 分)

1. 用户体验界面构建：开发易于操作的界面，包括上传图像区域和展示分析结果空间。
2. 初始化配置：系统启动时加载表情识别模型和情感分类标签，确保即时可用。
3. 用户内容提交：用户通过界面上传面部照片，作为表情分析素材。
4. 数据准备：系统自动处理照片，执行调整大小、裁剪和标准化等步骤。
5. 情绪检测：图像预处理后，系统利用模型进行表情识别。
6. 输出呈现：识别完成后，系统直观显示表情类别和标识符。
