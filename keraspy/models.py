import tensorflow as tf


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, w, h, name=None, **kwargs):
        super(ResizeLayer, self).__init__()

        self.w = w
        self.h = h

    def call(self, inputs):
        return tf.image.resize(inputs, (self.w, self.h))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'w': self.w,
            'h': self.h,
        })
        return config


def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', name="", relu=True):
    if(conv_type == 'ds'):
        x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides, name=f"{name}_SeparableConv")(inputs)
    else:
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides, name=f"{name}_Conv")(inputs)  

    x = tf.keras.layers.BatchNormalization(name=f"{name}_BatchNorm")(x)

    if (relu):
        x = tf.keras.activations.relu(x)

    return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False, index=""):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1), name=f"Residual_bottleneck_entry_{index}")

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', name=f"Residual_bottleneck_exit_{index}", relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n, index):
    x = _res_bottleneck(inputs, filters, kernel, t, strides, index=f"{index}_0")

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True, index=f"{index}_{i}")

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    # w = 64
    # h = 32
    w = 32
    h = 16

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = ResizeLayer(w, h)(x)
        # x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

        concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)


def get_model(size):
    # Input Layer
    input_layer = tf.keras.layers.Input(shape=(size[0], size[1], 3), name='input_layer')

    # Step 1: Learning to DownSample
    lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides=(2, 2), name="Downsampling_1")
    lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides=(2, 2), name="Downsampling_2")
    lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides=(2, 2), name="Downsampling_3")

    # Step 2: Global Feature Extractor
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3, index=0)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3, index=1)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3, index=2)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])

    # Step 3: Feature Fusion
    ff_layer1 = conv_block(lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), name="Feature_Fusion", relu=False)
    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1), activation=None, dilation_rate=(4, 4))(ff_layer2)

    # old approach with DepthWiseConv2d
    #ff_layer2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)

    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)

    # Step 4: Classifier
    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1), name='DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1), name='DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = conv_block(classifier, 'conv', 2, (1, 1), strides=(1, 1), padding='same', name="Final_reducing_convolution", relu=False)
    classifier = tf.keras.layers.Dropout(0.3)(classifier)

    classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
    classifier = tf.keras.activations.softmax(classifier)

    # classifier = tf.keras.layers.Reshape(size)(classifier)
    # classifier = tf.keras.activations.sigmoid(classifier)

    # Result
    fast_scnn = tf.keras.Model(inputs=input_layer, outputs=classifier, name='Fast_SCNN')
    return fast_scnn
