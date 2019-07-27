import os
import tensorflow as tf

path = '/Users/jbsnyder/Projects/ResNet/imagenet/train/'

os.listdir(path)

tdf = tf.data.Dataset.from_tensor_slices([os.path.join(path, i) for i in os.listdir(path)])

tdf = tdf.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=8,
            sloppy=True,
            prefetch_input_elements=16
        ))


feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }

def preprocess_image_record(record):
    obj = tf.parse_single_example(record, feature_map)
    imgdata = tf.image.resize(tf.image.decode_jpeg(obj['image/encoded']), (244, 244)) / 255.
    label = tf.one_hot(tf.cast(obj['image/class/label'], tf.int32), depth=1000)
    bbox = tf.stack([obj['image/object/bbox/%s' % x].values for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
    text = obj['image/class/text']
    return imgdata, label

tdf = tdf.map(preprocess_image_record).shuffle(32).repeat().batch(32)

def load_from_file(data_dir):
    '''
    Return training and validation data
    Parameters
    ----------
    data_dir

    Returns
    -------

    '''
    files = os.listdir(data_dir)
    train_files = [i for i in files if "01024" in i]
    train_files = [os.path.join(data_dir, i) for i in train_files]
    test_files = [i for i in files if "00128" in i]
    test_files = [os.path.join(data_dir, i) for i in test_files]
    train_tdf = tf.data.Dataset.from_tensor_slices(train_files)
    train_tdf = train_tdf.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=8,
            sloppy=True,
            prefetch_input_elements=16
        ))
    test_tdf = tf.data.Dataset.from_tensor_slices(test_files)
    test_tdf = test_tdf.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=8,
            sloppy=True,
            prefetch_input_elements=16
        ))
    train_tdf = train_tdf.map(preprocess_image_record)
    test_tdf = test_tdf.map(preprocess_image_record)
    return train_tdf, test_tdf

