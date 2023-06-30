import itertools

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.metrics import classification_report

# 1.数据准备
# 读取每个文件中的图片
from tensorflow.python.ops.confusion_matrix import confusion_matrix

imgs_path = glob.glob('birds/*/*.jpg')
# 获取标签的名称
all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]
# 把标签名称进行去重
labels_names = np.unique(all_labels_name)
# 包装为字典，将名称映射为序号
label_to_index = {name: i for i, name in enumerate(labels_names)}
# 反转字典
index_to_label = {v: k for k, v in label_to_index.items()}
# 将所有标签映射为序号
all_labels = [label_to_index[name] for name in all_labels_name]
# 将数据随机打乱，划分为训练数据和测试数据
np.random.seed(2023)
random_index = np.random.permutation(len(imgs_path))
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]
# 切片，取80%作为训练数据，20%作为测试数据
i = int(len(imgs_path) * 0.8)
train_path = imgs_path[:i]
train_labels = all_labels[:i]
test_path = imgs_path[i:]
test_labels = all_labels[i:]

# 2.数据集构建与预处理
# 构建数据集
train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))

# 读取图片路径并进行预处理
def load_img(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label
# 根据需求设置并行处理的线程数
AUTOTUNE = tf.data.experimental.AUTOTUNE
# 对训练数据和测试数据应用预处理函数和并行处理
train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUNE)
# 设置批量大小和缓冲区大小
BATCH_SIZE = 24
BUFFER_SIZE = 1000
# 打乱、分批次和预取数据
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(200)
])

# 输出模型信息
model.summary()

# 指定训练参数
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 定义保存模型的回调函数
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_model.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# 训练模型
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=test_ds,
    callbacks=[checkpoint_callback],
)

# 成功/失败率曲线
plt.plot(history.epoch, history.历史['accuracy'], label='accuracy')
plt.plot(history.epoch, history.历史['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(history.epoch, history.历史['loss'], label='loss')
plt.plot(history.epoch, history.历史['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.显示()


# 预测测试数据
y_pred = model.predict(test_ds)
# 将预测结果转换为类别序号
y_pred = np.argmax(y_pred, axis=1)
# 输出分类报告
print(classification_report(test_labels, y_pred, target_names=labels_names))
# 计算混淆矩阵
cm = confusion_matrix(test_labels, y_pred)
# 显示混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm。Blues)
plt.标题("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels_names))
plt.xticks(tick_marks, labels_names, rotation=45)
plt.yticks(tick_marks, labels_names)
thresh = cm.max() / 2。
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.显示()
