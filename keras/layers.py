import keras


def identity(inputs):
    return inputs


def relu6(inputs):
    return keras.layers.ReLU(max_value=6)(inputs)


def hardSigmoid(inputs):
    return relu6(inputs + 3.) / 6.


def hardSwish(inputs):
    return keras.layers.Lambda(lambda imgs:imgs*hardSigmoid(imgs))(inputs)

def squeeze(inputs):
    x = keras.backend.squeeze(inputs, 1)
    x = keras.backend.squeeze(x, 1)
    return x


def globalAveragePooling2D(inputs):
    pool_size = tuple(map(int, inputs.shape[1:3]))
    return keras.layers.AveragePooling2D(pool_size=pool_size)(inputs)


def batchNormalization(inputs):
    return keras.layers.BatchNormalization(momentum=0.99)(inputs)


def convNormAct(inputs, filters: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, norm_layer: str = None,
                act_layer: str = 'relu', use_bias: bool = True, l2_reg: float = 1e-5):
    if padding > 0:
        x = keras.layers.ZeroPadding2D(padding=padding)(inputs)
    else:
        x = identity(inputs)

    x = keras.layers.Conv2D(filters, kernel_size, strides=stride,
                            kernel_regularizer=keras.layers.regularizers.l2(l2_reg), use_bias=use_bias)(x)

    if norm_layer:
        x = batchNormalization(x)

    _available_activation = {
        'relu': keras.layers.ReLU(),
        'relu6': relu6,
        'hswish': hardSwish,
        'hsigmoid': hardSigmoid,
        'softmax': keras.layers.Softmax()
    }
    if act_layer in _available_activation.keys():
        x = _available_activation[act_layer](x)


    return x


def bneck(inputs, out_channels: int, exp_channels: int, kernel_size: int, stride: int, use_se: bool,
          act_layer: str, l2_reg: float = 1e-5):
    x = convNormAct(inputs, exp_channels, kernel_size=1, norm_layer='bn', act_layer=act_layer,
                    use_bias=False, l2_reg=l2_reg)

    dw_padding = (kernel_size - 1) // 2
    x = keras.layers.ZeroPadding2D(padding=dw_padding)(x)
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                                     depthwise_regularizer=keras.regularizers.l2(l2_reg),
                                     use_bias=False)(x)
    x = batchNormalization(x)
    if use_se:
        x = seBottleneck(x, l2_reg=l2_reg)

    if act_layer == 'relu':
        x = keras.layers.ReLU()(x)
    elif act_layer == 'hswish':
        x = hardSwish(x)
    else:
        x = identity(x)
    x = convNormAct(x, out_channels, kernel_size=1, norm_layer='bn', act_layer=None, use_bias=False,
                    l2_reg=l2_reg)

    if stride == 1 and int(inputs.shape[-1]) == out_channels:
        return keras.layers.add([inputs,x])
    else:
        return x


def seBottleneck(inputs, reduction: int = 4, l2_reg: float = 0.01):
    input_channels = int(inputs.shape[-1])
    x = globalAveragePooling2D(inputs)
    x = convNormAct(x, input_channels // reduction, kernel_size=1, norm_layer=None, act_layer='relu',
                    use_bias=False, l2_reg=l2_reg)
    x = convNormAct(x, input_channels, kernel_size=1, norm_layer='hsigmoid', use_bias=False,
                    l2_reg=l2_reg)
    return keras.layers.multiply([inputs,x])


def lastStage(inputs, penultimate_channels: int, last_channels: int, num_classes: int, l2_reg: float):
    conv1 = convNormAct(inputs, penultimate_channels, kernel_size=1, stride=1, norm_layer='bn', act_layer='hswish',
                        use_bias=False, l2_reg=l2_reg)
    gap = globalAveragePooling2D(conv1)
    conv2 = convNormAct(gap, last_channels, kernel_size=1, norm_layer=None, act_layer='hswish', l2_reg=l2_reg)
    dropout = keras.layers.Dropout(rate=0.2)(conv2)
    conv3 = convNormAct(dropout, num_classes, kernel_size=1, norm_layer=None, act_layer='softmax', l2_reg=l2_reg)
    # ret = squeeze(conv3)
    ret = keras.layers.Lambda(lambda imgs:keras.backend.reshape(imgs,(-1,int(conv3.shape[-1]))))(conv3)
    return ret
