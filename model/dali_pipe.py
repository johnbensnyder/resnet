import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
import nvidia.dali.plugin.tf as dali_tf

class TFRecordPipeline(Pipeline):
    def __init__(self, tfrecord_files, idx_files, 
                 batch_size, device_id=0, rank=0,
                 total_devices=1, num_threads=4):
        super(TFRecordPipeline, self).__init__(batch_size,
                                         num_threads,
                                         device_id)
        self.input = ops.TFRecordReader(path = tfrecord_files, index_path = idx_files,
                                        shard_id = rank, num_shards = total_devices,
                                        random_shuffle = True,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                         'image/class/label':         tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                         })
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", resize_shorter = 256)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16,
                                            crop = (224, 224),
                                            image_type = types.RGB,
                                            mean = [0, 0, 0],
                                            std = [1., 1., 1.],
                                            output_layout='HWC')
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.flip = ops.CoinFlip()
        self.brightness = ops.Uniform(range = (0.5, 1.5))
        self.contrast = ops.Uniform(range = (0.8, 1.3))
        self.cast = ops.Cast(device = "gpu", dtype = types.FLOAT16)
        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        images = self.decode(inputs["image/encoded"])
        resized_images = self.resize(images)
        resized_images = resized_images/255.
        output = self.cmnp(resized_images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform(), mirror=self.flip()) 
        return (output, self.cast(inputs["image/class/label"].gpu()))

    def iter_setup(self):
        pass

def dali_generator(tfrecord_files, idx_files, 
                 batch_size, num_threads=4, device_id=0, 
                 rank=0, total_devices=1):
    pipe = TFRecordPipeline(tfrecord_files, idx_files, 
                 batch_size, device_id, 
                 total_devices, num_threads)
    pipe.build()
    daliop = dali_tf.DALIIterator()
    while True:
        images, labels =  daliop(pipeline = pipe, 
                          shapes = [(batch_size, 224, 224, 3), ()], 
                          dtypes = [tf.float16, tf.float16])
        labels -= 1
        yield images, labels