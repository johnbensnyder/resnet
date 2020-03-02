from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model.dali_pipe import dali_generator
from model.lars import LARS
from model.scheduler import WarmupExponentialDecay, PiecewiseConstantDecay
import horovod.tensorflow as hvd
import tensorflow_addons as tfa
import argparse
from time import time
'''
mpirun -np 8 -H localhost:8 --bind-to none \
/home/ubuntu/anaconda3/envs/tensorflow2_p36/bin/python \
/home/ubuntu/shared_workspace/resnet/train.py \
--data_dir /home/ubuntu/shared_workspace/data/imagenet \
--index_dir /home/ubuntu/shared_workspace/data/imagenet_index \
--batch_size 4096 \
--num_epochs 70 \
--val_per_epoch


mpirun -np 32 -H \
--hostfile /home/ubuntu/shared_workspace/hosts \
-x FI_PROVIDER="efa" \
-x FI_EFA_TX_MIN_CREDITS=64 \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
-mca btl_vader_single_copy_mechanism none \
--mca btl tcp,self \
--mca btl_tcp_if_exclude lo,docker0 \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_DEBUG=INFO \
/home/ubuntu/anaconda3/envs/tensorflow2_p36/bin/python \
/home/ubuntu/shared_workspace/resnet/train.py \
--data_dir /home/ubuntu/shared_workspace/data/imagenet \
--index_dir /home/ubuntu/shared_workspace/data/imagenet_index \
--batch_size 16384 \
--num_epochs 90 \
--val_per_epoch
'''




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

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

@tf.function
def validation_step(images, labels, model, loss_func):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_5_pred = tf.math.top_k(pred, k=5)[1]
    top_1_pred = tf.math.top_k(pred, k=1)[1]
    labels = tf.cast(labels, tf.int32)
    # top_1 = sum([label in a_pred for label, a_pred in zip(labels, top_1_pred)])
    # top_5 = sum([label in a_pred for label, a_pred in zip(labels, top_5_pred)])
    return loss, top_1_pred, top_5_pred

def validation(validation_tdf, model, loss_func, per_gpu_batch, steps = 64):
    print("Running Validation")
    loss_tracker = []
    top_1_tracker = 0
    top_5_tracker = 0
    for _ in tqdm(range(steps)):
        images, labels = next(validation_tdf)
        loss, top_1_pred, top_5_pred = validation_step(images, labels, model, loss_func)
        loss_tracker.append(loss.numpy())
        labels = tf.cast(labels, tf.int32)
        top_1 = sum([label in a_pred for label, a_pred in zip(labels, top_1_pred)])
        top_5 = sum([label in a_pred for label, a_pred in zip(labels, top_5_pred)])
        top_1_tracker += top_1
        top_5_tracker += top_5
    return sum(loss_tracker)/len(loss_tracker), top_1_tracker/(steps*per_gpu_batch), top_5_tracker/(steps*per_gpu_batch)


@tf.function
def train_step(images, labels, model, optimizer, loss_func):
    with tf.GradientTape() as tape:
        pred = model(images, training=True)
        loss = loss_func(labels, pred)
        scaled_loss = optimizer.get_scaled_loss(loss)
    tape = hvd.DistributedGradientTape(tape)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    # grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_epoch(steps, train_tdf, model, optimizer, loss_func, scheduler, rank=0):
    if rank==0:
        #val_loss, top_1, top_5 = validation()
        #print("Validation loss: {} Top 1 {} Top 5 {}".format(val_loss, top_1, top_5))
        loss = []
        progressbar = tqdm(range(steps))
        for batch in progressbar:
            images, labels = next(train_tdf)
            loss.append(train_step(images, labels, model, optimizer, loss_func).numpy())
            progressbar.set_description("train loss: {0:.4f}, learning rate: {1:.4f}".format(np.array(loss[-100:]).mean(),
                                                                                             scheduler(optimizer.iterations)))
        #val_loss, top_1, top_5 = validation()
        #loss = np.array(loss).mean()
        #print("\ntrain_loss: {}\nvalidation_loss: {}\ntop_1: {}\ntop_5: {}".format(loss, val_loss, top_1, top_5))
        #return loss, val_loss, top_1, top_5
    else:
        for batch in range(steps):
            images, labels = next(train_tdf)
            _ = train_step(images, labels, model, optimizer, loss_func)

def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=False)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--' + name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    return cmdline

def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--data_dir',
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('--index_dir',
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=4096, type=int,
                         help="""Global batch size""")
    cmdline.add_argument('-n', '--num_epochs', default=70, type=int,
                         help="""Number of epochs""")
    add_bool_argument(cmdline, '--val_per_epoch', help="""Whether to run validation each epoch""")
    return cmdline

def main():
    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    data_dir = Path(FLAGS.data_dir)
    index_dir = Path(FLAGS.index_dir)
    train_files = [i.as_posix() for i in data_dir.glob('*1024')]
    train_index = [i.as_posix() for i in index_dir.glob('*1024')]
    val_files = [i.as_posix() for i in data_dir.glob('*0128')]
    val_index = [i.as_posix() for i in index_dir.glob('*0128')]

    num_epochs = FLAGS.num_epochs
    global_batch = FLAGS.batch_size
    per_gpu_batch = global_batch//hvd.size()
    image_count = 1282048
    steps_per_epoch = image_count//global_batch
    learning_rate = 0.001*global_batch/256
    scaled_rate = 0.1*global_batch/256

    #scheduler = WarmupExponentialDecay(learning_rate, 
    #                scaled_rate, steps_per_epoch*5, steps_per_epoch*num_epochs, 1e-8)
    scheduler = PiecewiseConstantDecay(learning_rate,
                        scaled_rate, steps_per_epoch*5, 
                        [steps_per_epoch*30, steps_per_epoch*60], [scaled_rate, scaled_rate*.1, scaled_rate*.01])
    train_tdf = dali_generator(train_files, train_index, per_gpu_batch, num_threads=8, 
                               device_id=hvd.local_rank(), rank=hvd.rank(), total_devices=hvd.size())
    validation_tdf = dali_generator(val_files, val_index, per_gpu_batch, num_threads=8, device_id=0, total_devices=1)
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)
    optimizer = LARS(scheduler, use_nesterov=False, clip=False)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    
    start_time = time()
    for epoch in range(num_epochs):
        if hvd.rank()==0:
            print("starting epoch: {}".format(epoch+1))
            train_epoch(steps_per_epoch, train_tdf, model, optimizer, loss_func, scheduler, hvd.rank())
            if FLAGS.val_per_epoch:
                val_loss, top_1, top_5 = validation(validation_tdf, model, loss_func, per_gpu_batch, steps = 64)
                print("\nval_loss {}\ntop_1 {}\ntop_5 {}".format(val_loss, top_1, top_5))
        else:
            train_epoch(steps_per_epoch, train_tdf, model, optimizer, loss_func, scheduler, hvd.rank())
    if hvd.rank()==0:
        running_time = time()-start_time
        val_loss, top_1, top_5 = validation(validation_tdf, model, loss_func, per_gpu_batch, steps = 256)
        print("\nval_loss {}\ntop_1 {}\ntop_5 {}".format(val_loss, top_1, top_5))
        with open("resnet_perf_1.txt", 'w') as outfile:
            outfile.write("time {}\nval loss {}\ntop 1 {}\ntop 5 {}\nbatch {}\nhvd size {}".format(running_time,
                                                                                        val_loss, top_1, top_5,
                                                                                              global_batch, hvd.size()))

if __name__ == '__main__':
    main()
