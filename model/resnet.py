import tensorflow as tf

class ResnetBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size, strides, identity):
        super(ResnetBlock, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters, 1 if strides==1 else 3, 
                                            strides, padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, 
                                            1, padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters*4, 1, 
                                            1, padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.layers.Conv2D(filters*4, 1, strides, padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())
        if identity:
            self.shortcut = tf.identity
    
    def call(self, X, training=True):
        shortcut = self.shortcut(X)
        X = self.conv0(X)
        X = self.bn0(X, training)
        X = tf.keras.activations.relu(X)
        X = self.conv1(X)
        X = self.bn1(X, training)
        X = tf.keras.activations.relu(X)
        X = self.conv2(X)
        X = self.bn2(X, training)
        X = tf.keras.activations.relu(X)
        X = tf.keras.layers.add([X, shortcut])
        X = tf.keras.activations.relu(X)
        return X

class ResnetStage(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size, strides, blocks):
        super(ResnetStage, self).__init__()
        self.conv_block = ResnetBlock(filters, kernel_size, strides, identity=False)
        self.blocks=blocks
        for block in range(blocks):
            self.__dict__['identity_{}'.format(block)] = ResnetBlock(filters, kernel_size, 1, identity=True)
    
    def call(self, X, training=True):
        X = self.conv_block(X, training)
        for block in range(self.blocks):
            X = self.__dict__['identity_{}'.format(block)](X)
        return X
    
class Resnet50(tf.keras.models.Model):
    
    def __init__(self, output_size=1000):
        super(Resnet50, self).__init__()
        # stage 0
        self.conv0 = tf.keras.layers.Conv2D(64, 7, 2)
        self.bn0 = tf.keras.layers.BatchNormalization()
        # stage 1
        self.stage1 = ResnetStage(64, 3, 1, 2)
        # stage 2
        self.stage2 = ResnetStage(128, 3, 2, 3)
        # stage 3
        self.stage3 = ResnetStage(256, 3, 2, 5)
        # stage 4
        self.stage4 = ResnetStage(512, 3, 2, 2)
        # output
        self.dense_out = tf.keras.layers.Dense(output_size, activation='softmax')
    
    def call(self, X, training=True):
        X = tf.keras.layers.ZeroPadding2D((3,3))(X)
        X = self.conv0(X)
        X = self.bn0(X, training)
        X = tf.keras.activations.relu(X)
        X = tf.keras.layers.MaxPool2D((3,3), (2,2))(X)
        X = self.stage1(X, training)
        X = self.stage2(X, training)
        X = self.stage3(X, training)
        X = self.stage4(X, training)
        X = tf.keras.layers.AveragePooling2D()(X)
        X = tf.keras.layers.Flatten()(X)
        X = self.dense_out(X)
        return X