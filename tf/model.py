from layers import *
import tensorflow_datasets as tfds
import numpy as np

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def build_mobilenet_small(inputs, num_classes: int = 1001,
                          width_multiplier: float = 1.0,
                          divisible_by: int = 8,
                          l2_reg: float = 1e-5, ):
    bneck_settings = [
        # k   exp   out  SE   NL  s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hswish", 2],
        [5, 240, 40, True, "hswish", 1],
        [5, 240, 40, True, "hswish", 1],
        [5, 120, 48, True, "hswish", 1],
        [5, 144, 48, True, "hswish", 1],
        [5, 288, 96, True, "hswish", 2],
        [5, 576, 96, True, "hswish", 1],
        [5, 576, 96, True, "hswish", 1],
    ]
    x = conv_bn_relu(inputs,
                     16,
                     kernel_size=3,
                     stride=2,
                     padding=1,
                     norm_layer="bn",
                     act_layer="hswish",
                     use_bias=False,
                     l2_reg=l2_reg
                     )

    for idx, (k, exp, out, SE, NL, s) in enumerate(bneck_settings):
        out_channels = _make_divisible(out * width_multiplier, divisible_by)
        exp_channels = _make_divisible(exp * width_multiplier, divisible_by)
        x = bneck(x,
                  out_channels=out_channels,
                  exp_channels=exp_channels,
                  kernel_size=k,
                  stride=s,
                  use_se=SE,
                  act_layer=NL, index=idx
                  )
    penultimate_channels = _make_divisible(960 * width_multiplier, divisible_by)
    last_channels = _make_divisible(1280 * width_multiplier, divisible_by)
    x = lastStage(x,
                  penultimate_channels,
                  last_channels,
                  num_classes,
                  l2_reg=l2_reg,
                  )
    return x

def build_model(num_classes = 100):
    inputs = tf.placeholder(dtype=tf.float32, shape=dataset_shape, name='input')
    y_true = tf.placeholder(dtype=tf.int64, shape=[None, ])
    y_hot = tf.one_hot(y_true, num_classes)

    y_pred = build_mobilenet_small(inputs, num_classes)
    y_pred_label = tf.argmax(y_pred, axis=-1, name='pred_labels')

    losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_hot, logits=y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(losses)

    return inputs, y_true, losses, train_op


def train():
    dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
    dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    it = dataset.make_one_shot_iterator()
    next_op = it.get_next()

    inputs, y_true, losses, train_op = build_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            features = sess.run(next_op)
            images, labels = features['image'], features['label']
            images = images/255. #int to float
            _, loss = sess.run([train_op, losses], feed_dict={inputs: images, y_true: labels})
            if i % 50 == 0:
                print('step', i, 'loss:', loss)
        saver.save(sess, 'models/' + dataset_name)


def test():
    dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
    dataset = dataset.shuffle(1024).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    it = dataset.make_one_shot_iterator()
    next_op = it.get_next()

    saver = tf.train.import_meta_graph('models/' + dataset_name + '.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('models'))

    inputs = sess.graph.get_tensor_by_name('input:0')
    outputs = sess.graph.get_tensor_by_name('pred_labels:0')
    total = 0
    correct = 0
    while True:
        try:
            features = sess.run(next_op)
            images, labels = features['image']/255., features['label']
            pred_labels = sess.run(outputs, feed_dict={inputs: images})
            total += labels.shape[0]
            correct += np.sum(np.cast(pred_labels == labels, np.float))

        except Exception:
            print('accuracy:', correct / total)
            break


dataset_name = 'cifar100'
dataset_shape = [None, 32, 32, 3]

if __name__ == '__main__':
    train()
    test()
