import tensorflow as tf
from model import ResNet50
from preprocess import load_from_file
tf.enable_eager_execution()

data_dir = '/home/ubuntu/data/tf-imagenet/'

train, test = load_from_file(data_dir)

model = ResNet50()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train = train.repeat().batch(256)
test = test.batch(256)

model.fit(train, epochs=10, steps_per_epoch=100)