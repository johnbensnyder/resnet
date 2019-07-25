"""
Creates an EC2 cluster for distributed deep learning and trains resnet50
"""

"""
Setup cluster name, number of instances and instance type
This is setup to use p3dn.24xlarge
"""

INSTANCE_TYPE = 'p3dn.24xlarge'
GPUS_PER_INSTANCE = 8
INTANCE_COUNT = 4
CLUSTER_NAME = 'jbsnyder-dl-cluster'
SSH_KEY = 'jbsnyder'
PEM_FILE = '~/.aws/jbsnyder.pem'
IAM_EC2_ROLE = 'jbsnyder'

from ec2_cluster import infra, control
import boto3
import json
from time import time

# launch a cluster
cluster = infra.EC2NodeCluster(INTANCE_COUNT, CLUSTER_NAME, 'us-east-1')

params = {'az': 'us-east-1b', 'vpc_id': 'vpc-f6570b8d', 'subnet_id': 'subnet-58b35b04',
          'ami_id': 'ami-0757fc5a639fe7666', 'ebs_snapshot_id': 'snap-0876a940246970049',
          'volume_gbs': 75, 'volume_type': 'gp2', 'key_name': SSH_KEY,
          'security_group_ids': ['sg-0d5ee02d5f0acb5f3'], 'iam_ec2_role_name': IAM_EC2_ROLE,
          'instance_type': INSTANCE_TYPE}

status = cluster.launch(**params) #TODO this sometimes throws an error "terminal failure state" but instance still starts

# save cluster info
cluster_info = {'cluster_name': CLUSTER_NAME, 'instance_type': INSTANCE_TYPE,
                'instance_count': INTANCE_COUNT, 'ssh_key': SSH_KEY,
                'instance_ids': cluster.instance_ids,
                'ip_addresses': cluster.ips}

with open('cluster_info_{}.json'.format(int(time()*1000)), 'w') as outfile:
    outfile.write(json.dumps(cluster_info))

# create and attach ebs volumes
client = boto3.client('ec2')

# create an ssh interace to cluster
cluster_ssh = control.ClusterShell('ubuntu', cluster.public_ips[0], cluster.public_ips[1:], PEM_FILE)

# mount all ebs volumes in the same place
cluster_ssh.run_on_all('sudo mkfs -t ext4 /dev/nvme2n1')
cluster_ssh.run_on_all('sudo mkdir ~/data')
cluster_ssh.run_on_all('sudo mount /dev/nvme2n1 ~/data')
cluster_ssh.run_on_all('sudo chmod 777 ~/data')

# create config file of workers
def worker_config(host, ip_address, port=22, user='ubuntu', identity_file=PEM_FILE):
    # format ip address
    address = 'ec2-{}.compute-1.amazonaws.com'.format('-'.join(ip_address.split('.')))
    formatted_string = """Host {}
    HostName {}
    port {}
    user {}
    IdentityFile {}
    IdentitiesOnly yes

""".format(host, address, port, user, identity_file)
    return formatted_string

with open("worker_config", "w") as outfile:
    for num, address in enumerate(cluster.ips['worker_public_ips']):
        outfile.writelines(worker_config('server{}'.format(num), address))

cluster_ssh.copy_from_local_to_master('/Users/jbsnyder/PycharmProjects/resnet/worker_config', '/home/ubuntu/.ssh/config')

cluster_ssh.run_on_all('mkdir ~/.aws')

cluster_ssh.copy_from_local_to_master('/Users/jbsnyder/.aws/jbsnyder.pem', '/home/ubuntu/.aws/jbsnyder.pem')

# create the hosts file for horovod
with open("hosts", "w") as outfile:
    outfile.writelines("localhost slots={}\n".format(GPUS_PER_INSTANCE))
    for worker in range(len(cluster.ips['worker_public_ips'])):
        outfile.writelines("server{} slots={}\n".format(worker, GPUS_PER_INSTANCE))

cluster_ssh.run_on_all('git clone https://github.com/johnbensnyder/resnet/')

cluster_ssh.copy_from_local_to_master('/Users/jbsnyder/PycharmProjects/resnet/hosts', '/home/ubuntu/resnet/hosts')

cluster_ssh.run_on_all('mkdir /home/ubuntu/model')
cluster_ssh.run_on_all('mkdir /home/ubuntu/results')
cluster_ssh.run_on_all('mkdir /home/ubuntu/summaries')
cluster_ssh.run_on_all('mkdir /home/ubuntu/data/tf-imagenet')

cluster_ssh.copy_from_local_to_all('/Users/jbsnyder/.aws/config', '/home/ubuntu/.aws/config')
cluster_ssh.copy_from_local_to_all('/Users/jbsnyder/.aws/credentials', '/home/ubuntu/.aws/credentials')

cluster_ssh = control.ClusterShell('ubuntu', cluster.public_ips[0], cluster.public_ips[1:], PEM_FILE)
cluster_ssh.run_on_all('aws s3 cp s3://aws-tensorflow-benchmarking/imagenet-armand/train-480px/ /home/ubuntu/data/tf-imagenet/ --recursive --quiet')
cluster_ssh.run_on_all('aws s3 cp s3://aws-tensorflow-benchmarking/imagenet-armand/validation-480px/ /home/ubuntu/data/tf-imagenet/ --recursive --quiet')

cluster_ssh.run_on_all('source activate tensorflow_p36 && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali')
cluster_ssh.run_on_all('source activate tensorflow_p36 && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali-tf-plugin')

print(cluster.ips)

cluster_ssh.run_on_master('git clone https://github.com/horovod/horovod/')

cluster_ssh.run_on_master('source activate tensorflow_p36 && mpiexec --allow-run-as-root --bind-to socket -np 32 -hostfile /home/ubuntu/resnet/hosts python /home/ubuntu/resnet/RN50v1.5/train_and_eval.py')

cluster.terminate()

"""# Training
~/anaconda3/envs/tensorflow_p36/bin/mpirun -np 32 -hostfile /home/ubuntu/resnet/hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore /home/ubuntu/resnet/RN50v1.5/train_and_eval.py
	
	"""