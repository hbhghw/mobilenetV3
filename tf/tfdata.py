import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def one_hot(array,n_classes=None):
    if n_classes is None:
        n_classes = max(array)+1
    new_array = np.zeros((len(array),n_classes))
    new_array[list(range(len(array))),array] = 1
    return new_array

def dataGenerator(default_graph):
    dataset = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    it = dataset.make_one_shot_iterator()
    next_op = it.get_next()
    with tf.Session(graph=default_graph) as sess:
        while True:
            features = sess.run(next_op)
            images, labels = features['image'], features['label']
            yield images, one_hot(labels,10)

def test1():
    graph = tf.get_default_graph()
    ge = dataGenerator(graph)
    while True:
        images,labels = next(ge)
        print(images.shape,labels.shape)

def dataGenerator2():
    tf.enable_eager_execution()
    dataset = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    for features in dataset:  # 1*batch
        images, labels = features["image"], features["label"]
        # print(images.numpy(),labels.numpy())
        yield images.numpy(),labels.numpy()

def test2():
    ge2 = dataGenerator2()
    while True:
        images, labels = next(ge2)
        print(labels)
test1()
# test2()