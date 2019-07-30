import tensorflow as tf
from model import ResNet50
from preprocess import load_from_file
#tf.enable_eager_execution()

data_dir = '/home/ubuntu/model/data/tf-imagenet/'

train, test = load_from_file(data_dir)

model = ResNet50()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train = train.shuffle(5000).repeat().batch(128)
test = test.batch(128)

estimator = tf.keras.estimator.model_to_estimator(model)

model.fit(train, epochs=10, steps_per_epoch=16000)

def input_fn(path):
    train, test = load_from_file(data_dir)
    train = train.shuffle(5000).repeat(40).batch(128).prefetch(2)
    return train

estimator.train(lambda: input_fn(data_dir))

