import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np
import glob  

# 读取每个文件中的图片
imgs_path = glob.glob('birds/*/*.jpg')
# 获取标签的名称
all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]
# 把标签名称进行去重
labels_names = np.unique(all_labels_name)
# print(len(labels_names))
# 包装为字典,将名称映射为序号
label_to_index = dict((name, i) for i, name in enumerate(labels_names))
# print(label_to_index)
# 反转字典
index_to_label = dict((v, k) for k, v in label_to_index.items())
# print(index_to_label)
# 吧所有标签映射为序号
all_labels = [label_to_index.get(name) for name in all_labels_name]
# print(all_labels[:3])
# 将数据随机打乱，划分为训练数据及测试数据
np.random.seed(2021)  # 设置随机因子,伪随机，方便测试
random_index = np.random.permutation(len(imgs_path))
# 把图片路径和标签进行相同次序的打乱
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]
# 切片,取80%作为训练数据，20%作为测试数据
i = int(len(imgs_path) * 0.8)
# 训练数据
train_path = imgs_path[:i]
train_labels = all_labels[:i]
# 测试数据
test_path = imgs_path[i:]
test_labels = all_labels[i:]
# 编写数据集
train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))


# 读取图片路径并做预处理
def load_img(path, label):
    # 读取图片
    image = tf.io.read_file(path)
    # 解码
    image = tf.image.decode_jpeg(image, channels=3)
    # 重构图片大小
    image = tf.image.resize(image, [256, 256])
    # 转换数据类型
    image = tf.cast(image, tf.float32)
    # 归一化
    image = image / 255
    return image, label


# 根据需求读取线程数
AUTOTUEN = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUEN)
test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUEN)
print(train_ds)
# 图片一次加载32张
BATCH_SIZE = 32
# 缓存区设置300
train_ds = train_ds.repeat().shuffle(300).batch(BATCH_SIZE)
print("**缓存区设置300***")
print(train_ds)
test_ds = test_ds.batch(BATCH_SIZE)
# 建立模型
model = tf.keras.Sequential([
    # 卷积层，该卷积层输出64个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出64个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 池化层
    tf.keras.layers.MaxPooling2D(),
    # 卷积层，该卷积层输出128个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出128个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 池化层
    tf.keras.layers.MaxPooling2D(),
    # 卷积层，该卷积层输出256个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出256个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 池化层
    tf.keras.layers.MaxPooling2D(),
    # 卷积层，该卷积层输出512个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出512个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 池化层
    tf.keras.layers.MaxPooling2D(),
    # 卷积层，该卷积层输出512个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出512个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # 卷积层，该卷积层输出512个通道，卷积核为3*3，激活函数relu
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    #拉伸
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #分类为200
    tf.keras.layers.Dense(200)
])
#输出模型信息
model.summary()
#指明训练参数
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc']
              )
# 训练图片数量
train_count = len(train_path)
# 测试图片数量
test_count = len(test_path)
# 训练轮数
steps_per_epoch = train_count // BATCH_SIZE
print(steps_per_epoch)
# 测试轮数
validation_steps = test_count // BATCH_SIZE
print(validation_steps)

# 训练过程
print("训练过程")
history = model.fit(
    x=train_ds, epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps
)
print("可视化训练过程")
# 可视化训练过程
history.history.keys()

# 绘制成功率
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
# 绘制失败率
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()

# 记录准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["acc"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_acc"]

