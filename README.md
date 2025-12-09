# CIFAR-10 图像分类系统 - CNN Image Classifier

基于卷积神经网络 (CNN) 的 CIFAR-10 数据集图像分类系统，使用 TensorFlow/Keras 实现。

## 功能特性
- **CNN架构**: 三层卷积 + 池化 + 全连接
- **数据可视化**: 展示数据集样本图像
- **模型训练**: 完整的训练流程和验证
- **性能评估**: 训练/验证准确率对比
- **过拟合分析**: 计算和可视化训练-验证差距
- **模型保存**: 保存训练好的模型供后续使用

## 数据集信息 (CIFAR-10)
- **名称**: CIFAR-10 (Canadian Institute For Advanced Research)
- **样本数**: 60,000 (50,000训练 + 10,000测试)
- **图像尺寸**: 32×32 像素
- **颜色通道**: RGB (3通道)
- **类别数**: 10
- **类别分布**: 每个类别6,000张图像

### 类别标签
| 标签 | 英文 | 中文 |
|------|------|------|
| 0 | airplane | 飞机 |
| 1 | automobile | 汽车 |
| 2 | bird | 鸟 |
| 3 | cat | 猫 |
| 4 | deer | 鹿 |
| 5 | dog | 狗 |
| 6 | frog | 青蛙 |
| 7 | horse | 马 |
| 8 | ship | 船 |
| 9 | truck | 卡车 |

### 安装依赖
```bash
pip install -r requirements.txt
