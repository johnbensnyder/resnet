from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model.dali_pipe import dali_generator
from model.resnet import Resnet50
from model.lars import LARS
from model.scheduler import WarmupExponentialDecay
import horovod.tensorflow as hvd
import tensorflow_addons as tfa

hvd.init()

data_dir = Path('/workspace/shared_workspace/data/imagenet/')
index_dir = Path('/workspace/shared_workspace/data/imagenet_index/')
train_files = [i.as_posix() for i in data_dir.glob('*1024')]
train_index = [i.as_posix() for i in index_dir.glob('*1024')]
validation_files = [i.as_posix() for i in data_dir.glob('*0128')]
validation_index = [i.as_posix() for i in index_dir.glob('*0128')]

num_epochs = 10
batch_size = 128
global_batch = batch_size*hvd.size()
image_count = 1282048
steps_per_epoch = image_count//global_batch
learning_rate = 0.01*global_batch/256
scaled_rate = 0.4

tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-4)
tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
# mpirun -np 4 -H localhost:4 --bind-to none --allow-run-as-root python train.py

scheduler = WarmupExponentialDecay(tf.cast(learning_rate, tf.float16), scaled_rate, steps_per_epoch//10, steps_per_epoch*num_epochs, 0.001)
train_tdf = dali_generator(train_files, train_index, batch_size, num_threads=4, device_id=hvd.rank(), total_devices=hvd.size())
validation_tdf = dali_generator(validation_files, validation_index, batch_size, num_threads=4, device_id=hvd.rank(), total_devices=hvd.size())

model = Resnet50()
optimizer = LARS(scheduler, use_nesterov=False, clip=False)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

def validation_step(images, labels):
    with tf.device('/gpu:0'):
        pred = model(images, training=False)
        loss = loss_func(labels, pred)
        top_5_pred = tf.math.top_k(pred, k=5)[1]
        top_1_pred = tf.math.top_k(pred, k=1)[1]
        labels = tf.cast(labels, tf.int32)
    top_1 = sum([label in a_pred for label, a_pred in zip(labels, top_1_pred)])
    top_5 = sum([label in a_pred for label, a_pred in zip(labels, top_5_pred)])
    return loss, top_1, top_5

def validation(steps = 128):
    loss_tracker = []
    top_1_tracker = 0
    top_5_tracker = 0
    for _ in range(steps):
        images, labels = next(validation_tdf)
        loss, top_1, top_5 = validation_step(images, labels)
        loss_tracker.append(loss.numpy())
        top_1_tracker += top_1
        top_5_tracker += top_5
    return sum(loss_tracker)/len(loss_tracker), top_1_tracker/(steps*batch_size), top_5_tracker/(steps*batch_size)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred = model(images, training=True)
        loss = loss_func(labels, pred)
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_epoch(steps, rank=0):
    if rank==0:
        loss = []
        progressbar = tqdm(range(steps))
        for batch in progressbar:
            images, labels = next(train_tdf)
            loss.append(train_step(images, labels).numpy())
            progressbar.set_description("train loss: {0:.4f}, learning rate: {1:.4f}".format(np.array(loss[-100:]).mean(),
                                                                                             scheduler(optimizer.iterations)))
        val_loss, top_1, top_5 = validation()
        loss = np.array(loss).mean()
        print("\ntrain_loss: {}\nvalidation_loss: {}\ntop_1: {}\ntop_5: {}".format(loss, val_loss, top_1, top_5))
    else:
        for batch in range(steps):
            images, labels = next(train_tdf)
            _ = train_step(images, labels).numpy()

if hvd.rank()==0:
    loss, top_1, top_5 = validation(steps=128)
    print("Starting Values")
    print("\nloss: {}\ntop_1: {}\ntop_5: {}".format(loss, top_1, top_5))

if __name__=='__main__':
    for epoch in range(num_epochs):
        if hvd.rank()==0:
            print("starting epoch: {}".format(epoch))
        train_epoch(steps_per_epoch, hvd.rank())