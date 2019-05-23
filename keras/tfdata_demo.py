import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

tf.enable_eager_execution()
# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
# dataset = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
#
# # Build your input pipeline
# dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
#
# for features in dataset.take(-1):  # -1 means all
#     image, label = features["image"], features["label"]
#     print(image.shape)
#     cv2.imshow('1', image[0].numpy())
#     cv2.waitKey(1000)
#     print(label.numpy())

dataset = tfds.load(name="cifar10", split=tfds.Split.TRAIN)

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
dataset = tfds.as_numpy(dataset)


def data_generator():
    for features in dataset:
        yield features['image'], features['label']

dg = data_generator()
while True:  # 1*batch
    image, label = next(dg)
    print(image.shape)
    cv2.imshow('1', image[0])
    cv2.waitKey(1000)
    print(label)
