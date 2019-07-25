INSTANCE_TYPE = 'p3.2xlarge'
GPUS_PER_INSTANCE = 1
INTANCE_COUNT = 1
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

volume = client.create_volume(AvailabilityZone=params['az'],
                     Size=500,
                     VolumeType='gp2',
                     TagSpecifications=[{'ResourceType':'volume',
                                         'Tags':[{'Key':'Name',
                                                  'Value':'{}_ebs'.format(CLUSTER_NAME)}]}])

cluster.ec2_resource.Instance(cluster.instance_ids[0]).attach_volume(Device='/dev/sdh',
                                                                     VolumeId=volume['VolumeId'])

cluster_ssh = control.ClusterShell('ubuntu', cluster.public_ips[0], cluster.public_ips[1:], PEM_FILE)

cluster_ssh.run_on_all('sudo mkfs -t ext4 /dev/xvdh')
cluster_ssh.run_on_all('sudo mkdir ~/data')
cluster_ssh.run_on_all('sudo mount /dev/xvdh ~/data')
cluster_ssh.run_on_all('sudo chmod 777 ~/data')

cluster_ssh.run_on_all('mkdir ~/.aws')

cluster_ssh.run_on_all('git clone https://github.com/johnbensnyder/resnet/')

cluster_ssh.run_on_all('cd resnet && git pull')

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

cluster_ssh.run_on_master('/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python /home/ubuntu/resnet/RN50v1.5/train.py')