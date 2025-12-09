"""导入库"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf

# 强制 TkAgg 后端
matplotlib.use('TkAgg')
# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 忽略警告
warnings.filterwarnings('ignore')

# 加载cifar-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 归一化处理
x_train = x_train / 255
x_test = x_test / 255

# 打印测试集第8张图像的像素
print(f'测试集第8张图的像素：{x_test[7].shape}')
# 打印测试集的标签的形状
print(f'测试集的标签的形状：{y_test.shape}')

# 创建类名称列表
class_name = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
# 设置画板大小
plt.figure(figsize=(5, 5))

# 展示前9张图像
for i in range(9):
    # 创建子图
    plt.subplot(3, 3, i + 1)
    # 关闭横纵刻度
    plt.xticks([])
    plt.yticks([])
    # 添加x轴标签
    plt.xlabel(class_name[y_train[i][0]])
    # 设置显示图像
    plt.imshow(x_train[i])
plt.show(block=False)
plt.pause(3)
plt.close()

# 创建神经网络模型
model = tf.keras.models.Sequential()

# 创建第一卷积块
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(2))

# 创建第二卷积块
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

# 创建第三卷积块
model.add(tf.keras.layers.Conv2D(128, 3,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

# 添加平展层
model.add(tf.keras.layers.Flatten())
# 添加全连接层
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 添加dropout层
model.add(tf.keras.layers.Dropout(0.3))
# 添加输出层
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# 训练模型
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 获取差值
acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = acc - val_acc
print(f'差值：{gap:.2%}')

# 评估模型
_, test_acc = model.evaluate(x_test, y_test)
print(f'准确率：{test_acc:.2%}')