import tensorflow as tf
from collections import OrderedDict

class IdentityBlock(tf.keras.Model):

    def __init__(self, shape, filters):
        super(IdentityBlock, self).__init__()
        self.shape = shape
        self.filters = filters
        self.net = OrderedDict()
        self.net['conv_1'] = tf.keras.layers.Conv2D(filters=self.filters[0],
                                                    kernel_size=(1,1),
                                                    strides=(1,1),
                                                    padding='valid',
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_1'] = tf.keras.layers.BatchNormalization()
        self.net['activation_1'] = tf.keras.layers.ReLU()
        self.net['conv_2'] = tf.keras.layers.Conv2D(filters=self.filters[1],
                                                    kernel_size=(shape, shape),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_2'] = tf.keras.layers.BatchNormalization()
        self.net['activation_2'] = tf.keras.layers.ReLU()
        self.net['conv_3'] = tf.keras.layers.Conv2D(filters=self.filters[2],
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding='valid',
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_3'] = tf.keras.layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        for a_layer in self.net:
            x = self.net[a_layer](x)
        x = shortcut + x
        return tf.keras.activations.relu(x)

class ConvolutionBlock(tf.keras.Model):

    def __init__(self, shape, filters, stride):
        super(ConvolutionBlock, self).__init__()
        self.shape = shape
        self.filters = filters
        self.stride = stride
        self.net = OrderedDict()
        self.shortcut_path = OrderedDict()
        self.net['conv_1'] = tf.keras.layers.Conv2D(filters=self.filters[0],
                                                    kernel_size=(1, 1),
                                                    strides=(self.stride, self.stride),
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_1'] = tf.keras.layers.BatchNormalization()
        self.net['activation_1'] = tf.keras.layers.ReLU()
        self.net['conv_2'] = tf.keras.layers.Conv2D(filters=self.filters[1],
                                                    kernel_size=(shape, shape),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_2'] = tf.keras.layers.BatchNormalization()
        self.net['activation_2'] = tf.keras.layers.ReLU()
        self.net['conv_3'] = tf.keras.layers.Conv2D(filters=self.filters[2],
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_3'] = tf.keras.layers.BatchNormalization()
        self.shortcut_path['conv'] = tf.keras.layers.Conv2D(filters=self.filters[2],
                                                            kernel_size=(1,1),
                                                            strides=(self.stride, self.stride),
                                                            padding='valid',
                                                            kernel_initializer=tf.glorot_uniform_initializer())
        self.shortcut_path['batch_norm'] = tf.keras.layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        for a_layer in self.net:
            x = self.net[a_layer](x)
        for a_layer in self.shortcut_path:
            shortcut = self.shortcut_path[a_layer](shortcut)
        x = x + shortcut
        return tf.keras.activations.relu(x)

class ResNet50(tf.keras.Model):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.net = OrderedDict()
        self.net['zero_pad'] = tf.keras.layers.ZeroPadding2D((3,3))
        self.net['conv_init'] = tf.keras.layers.Conv2D(filters=64,
                                                       kernel_size=(7, 7),
                                                       strides=(2,2),
                                                       kernel_initializer=tf.glorot_uniform_initializer())
        self.net['batch_norm_init'] = tf.keras.layers.BatchNormalization()
        self.net['activation_init'] = tf.keras.layers.ReLU()
        self.net['max_pool'] = tf.keras.layers.MaxPool2D((3,3), strides=(2,2))
        # block 1
        self.net['conv_block_1'] = ConvolutionBlock(shape=3, filters=[64, 64, 256], stride=1)
        self.net['identity_block_1a'] = IdentityBlock(shape=3, filters=[64, 64, 256])
        self.net['identity_block_1b'] = IdentityBlock(shape=3, filters=[64, 64, 256])
        # block 2
        self.net['conv_block_2'] = ConvolutionBlock(shape=3, filters=[128, 128, 512], stride=2)
        self.net['identity_block_2a'] = IdentityBlock(shape=3, filters=[128, 128, 512])
        self.net['identity_block_2b'] = IdentityBlock(shape=3, filters=[128, 128, 512])
        self.net['identity_block_2c'] = IdentityBlock(shape=3, filters=[128, 128, 512])
        # block 3
        self.net['conv_block_3'] = ConvolutionBlock(shape=3, filters=[256, 256, 1024], stride=2)
        self.net['identity_block_3a'] = IdentityBlock(shape=3, filters=[256, 256, 1024])
        self.net['identity_block_3b'] = IdentityBlock(shape=3, filters=[256, 256, 1024])
        self.net['identity_block_3c'] = IdentityBlock(shape=3, filters=[256, 256, 1024])
        self.net['identity_block_3d'] = IdentityBlock(shape=3, filters=[256, 256, 1024])
        self.net['identity_block_3e'] = IdentityBlock(shape=3, filters=[256, 256, 1024])
        # block 4
        self.net['conv_block_4'] = ConvolutionBlock(shape=3, filters=[512, 512, 2048], stride=2)
        self.net['identity_block_4a'] = IdentityBlock(shape=3, filters=[512, 512, 2048])
        self.net['identity_block_4b'] = IdentityBlock(shape=3, filters=[512, 512, 2048])
        # avg pool
        self.net['avg_pool'] = tf.keras.layers.GlobalAveragePooling2D()
        # flatten
        self.net['flatten'] = tf.keras.layers.Flatten()
        # output
        self.net['output'] = tf.keras.layers.Dense(1000, activation='softmax',
                                                   kernel_initializer=tf.glorot_uniform_initializer())

    def call(self, x):
        for a_layer in self.net:
            x = self.net[a_layer](x)
        return x


