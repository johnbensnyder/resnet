from pathlib import Path 
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time
from model.dali_pipe import dali_generator
from model.resnet import Resnet50
from model.lars import LARS
from model.scheduler import WarmupExponentialDecay, PiecewiseConstantDecay
import horovod.tensorflow as hvd
#import tensorflow_addons as tfa
# mpirun -np 8 --hostfile /workspace/shared_workspace/hosts --bind-to none --allow-run-as-root python train.py
'''
mpirun -np 32 --hostfile /home/ubuntu/shared_workspace/hosts \
-x FI_PROVIDER="efa" \
-x FI_EFA_TX_MIN_CREDITS=64 \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
-mca btl_vader_single_copy_mechanism none \
--mca btl tcp,self \
--mca btl_tcp_if_exclude lo,docker0 \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_DEBUG=INFO \
/home/ubuntu/anaconda3/envs/tensorflow2_p36/bin/python /home/ubuntu/shared_workspace/resnet/train.py

mpirun -np 32 --hostfile /home/ubuntu/shared_workspace/hosts \
-x FI_PROVIDER="sockets" \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
-mca btl_vader_single_copy_mechanism none \
--mca btl tcp,self \
--mca btl_tcp_if_exclude lo,docker0 \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_DEBUG=INFO \
/home/ubuntu/anaconda3/envs/tensorflow2_p36/bin/python /home/ubuntu/shared_workspace/resnet/train.py

'''
hvd.init()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

data_dir = Path('/home/ubuntu/shared_workspace/data/imagenet/')
index_dir = Path('/home/ubuntu/shared_workspace/data/imagenet_index/')
train_files = [i.as_posix() for i in data_dir.glob('*1024')]
train_index = [i.as_posix() for i in index_dir.glob('*1024')]

global_batch = 16384
per_gpu_batch = global_batch//hvd.size()
image_count = 1282048
steps_per_epoch = image_count//global_batch
learning_rate = 0.01*global_batch/256
scaled_rate = 0.1*(global_batch/256)
num_epochs = 70

tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-4)
tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

scheduler = WarmupExponentialDecay(learning_rate, 
                                   scaled_rate, steps_per_epoch, steps_per_epoch*num_epochs, 1e-8)
#scheduler = PiecewiseConstantDecay(learning_rate, scaled_rate, steps_per_epoch,
#                                   [steps_per_epoch*3, steps_per_epoch*10, steps_per_epoch*30, steps_per_epoch*50],
#                                   [scaled_rate, scaled_rate*0.1, scaled_rate*0.01, scaled_rate*0.001, scaled_rate*0.0001])
train_tdf = dali_generator(train_files, train_index, per_gpu_batch, num_threads=8, 
                           device_id=hvd.local_rank(), rank=hvd.rank(), total_devices=hvd.size())
validation_tdf = dali_generator(train_files, train_index, per_gpu_batch, num_threads=8, device_id=0, total_devices=1)

model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)
optimizer = LARS(scheduler, use_nesterov=False, clip=False)
#optimizer = tfa.optimizers.SGDW(0.0001, learning_rate=scheduler, momentum=0.9, nesterov=True)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def validation_step(images, labels):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_5_pred = tf.math.top_k(pred, k=5)[1]
    top_1_pred = tf.math.top_k(pred, k=1)[1]
    labels = tf.cast(labels, tf.int32)
    # top_1 = sum([label in a_pred for label, a_pred in zip(labels, top_1_pred)])
    # top_5 = sum([label in a_pred for label, a_pred in zip(labels, top_5_pred)])
    return loss, top_1_pred, top_5_pred

def validation(steps = 64):
    loss_tracker = []
    top_1_tracker = 0
    top_5_tracker = 0
    for _ in range(steps):
        images, labels = next(validation_tdf)
        loss, top_1_pred, top_5_pred = validation_step(images, labels)
        loss_tracker.append(loss.numpy())
        labels = tf.cast(labels, tf.int32)
        top_1 = sum([label in a_pred for label, a_pred in zip(labels, top_1_pred)])
        top_5 = sum([label in a_pred for label, a_pred in zip(labels, top_5_pred)])
        top_1_tracker += top_1
        top_5_tracker += top_5
    return sum(loss_tracker)/len(loss_tracker), top_1_tracker/(steps*per_gpu_batch), top_5_tracker/(steps*per_gpu_batch)


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
        #val_loss, top_1, top_5 = validation()
        #print("Validation loss: {} Top 1 {} Top 5 {}".format(val_loss, top_1, top_5))
        loss = []
        progressbar = tqdm(range(steps))
        for batch in progressbar:
            images, labels = next(train_tdf)
            loss.append(train_step(images, labels).numpy())
            progressbar.set_description("train loss: {0:.4f}, learning rate: {1:.4f}".format(np.array(loss[-100:]).mean(),
                                                                                             scheduler(optimizer.iterations)))
        #val_loss, top_1, top_5 = validation()
        #loss = np.array(loss).mean()
        #print("\ntrain_loss: {}\nvalidation_loss: {}\ntop_1: {}\ntop_5: {}".format(loss, val_loss, top_1, top_5))
        #return loss, val_loss, top_1, top_5
    else:
        for batch in range(steps):
            images, labels = next(train_tdf)
            _ = train_step(images, labels)

'''if hvd.rank()==0:
    loss, top_1, top_5 = validation(steps=128)
    print("Starting Values")
    print("\nloss: {}\ntop_1: {}\ntop_5: {}".format(loss, top_1, top_5))
'''

if __name__=='__main__':
    start_time = time()
    top_1 = 0
    epoch = 1
    #for epoch in range(num_epochs):
    for epoch in range(num_epochs):
        if hvd.rank()==0:
            print("starting epoch: {}".format(epoch))
            train_epoch(steps_per_epoch, hvd.rank())
        else:
            train_epoch(steps_per_epoch, hvd.rank())
    if hvd.rank()==0:
        with open("resnet_perf_1.txt", 'w') as outfile:
            val_loss, top_1, top_5 = validation(steps = 256)
            outfile.write("time {}\nval loss {}\ntop 1 {}\ntop 5 {}\nbatch {}".format(time()-start_time,
                                                                                        val_loss, top_1, top_5,
                                                                                              global_batch))
