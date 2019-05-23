import keras
import tensorflow_datasets as tfds
from layers import *
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

def one_hot(array,n_classes=None):
    if n_classes is None:
        n_classes = max(array)+1
    new_array = np.zeros((len(array),n_classes))
    new_array[list(range(len(array))),array] = 1
    return new_array


def dataGenerator(deault_graph):  # data iterator
    with tf.Session(graph=deault_graph) as sess:
        dataset = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
        dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        it = dataset.make_one_shot_iterator()
        next_op = it.get_next()
        while True:
            features = sess.run(next_op)
            images, labels = features['image'], features['label']
            yield images,one_hot(labels,10)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def build_mobilnet_small(input_shape=(224, 224, 3),
                        num_classes: int = 10,
                         width_multiplier: float = 1.0,
                         l2_reg: float = 1e-5,divisible_by: int = 8 ):
    bneck_settings = [
        # k   exp   out  SE  NL s
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
        [5, 576, 96, True, "hswish", 1]
    ]

    inputs = keras.Input(shape=input_shape)
    #first layer
    x = convNormAct(inputs,
                    16,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer="bn",
                    act_layer="hswish",
                    use_bias=False,
                    l2_reg=l2_reg)

    #bneck layers
    for idx, (k, exp, out, SE, NL, s) in enumerate(bneck_settings):
        out_channels = _make_divisible(out * width_multiplier, divisible_by)
        exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

        x = bneck(x,
                  out_channels=out_channels,
                  exp_channels=exp_channels,
                  kernel_size=k,
                  stride=s,
                  use_se=SE,
                  act_layer=NL
                  )

    #last layer
    penultimate_channels = _make_divisible(960 * width_multiplier, divisible_by)
    last_channels = _make_divisible(1280 * width_multiplier, divisible_by)
    outputs = lastStage(x,
                  penultimate_channels,
                  last_channels,
                  num_classes,
                  l2_reg=l2_reg,
                  )

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def train(args):
    model = build_mobilnet_small(
        input_shape=(args.height, args.width, args.channels),
        num_classes=args.num_classes,
        width_multiplier=args.width_multiplier,
        l2_reg=args.l2_reg
    )
    default_graph = tf.get_default_graph()
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = keras.callbacks.TensorBoard(log_dir=args.logdir)
    model.fit_generator(dataGenerator(default_graph), steps_per_epoch=200,
                        epochs=args.num_epoch,
                        callbacks=[callbacks]
                        )

    model.save_weights(f"mobilenetv3_small_cifar10_{args.num_epoch}.h5")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--channels', type=int, default=3)

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--width_multiplier', type=float, default=1.0)
    parser.add_argument('--logdir', type=str, default='../logs')

    args = parser.parse_args()
    train(args)
