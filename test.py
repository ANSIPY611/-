import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow import keras
from keras import layers
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
#
imgs_path = glob.glob('birds/000.test/*.jpg')
# print(imgs_path)
# path='birds/167.Hooded_Warbler\Hooded_Warbler_0068_164872.jpg'

img_height = 256
img_width = 256
#读取模型
model = tf.keras.models.load_model("models/cnn_bird.h5")
for path in imgs_path:
    data = keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
    data = keras.preprocessing.image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = np.vstack([data])
    result = np.argmax(model.predict(data))
    print(path)
    print(result)
    print(index_to_label[result])
    
