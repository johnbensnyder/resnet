from subprocess import call
from pathlib import Path

tfrecord2idx_script = '/home/ubuntu/shared_workspace/resnet/model/tfrecord2idx'
data_dir = Path('/home/ubuntu/shared_workspace/data/imagenet/')
index_dir = Path('/home/ubuntu/shared_workspace/data/imagenet_index/')
files = data_dir.glob('*')

for file in files:
    call([tfrecord2idx_script, file, index_dir.joinpath(file.name).as_posix()])
