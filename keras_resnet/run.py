import tensorflow as tf
from model import ResNet50
from preprocess import load_from_file

data_dir = '/home/ubuntu/data/tf-imagenet/'

train, test = load_from_file(data_dir)

model = ResNet50()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train = train.shuffle(128).repeat().batch(128)
test = test.shuffle(128).repeat().batch(128)

model.fit(train, epochs=10, steps_per_epoch=100)