"""
Creates an EC2 cluster for distributed deep learning and trains resnet50
"""

"""
Setup cluster name, number of instances and instance type
This is setup to use p3dn.24xlarge
"""

INSTANCE_TYPE = 'p3dn.24xlarge'
GPUS_PER_INSTANCE = 8
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

cluster_ssh.run_on_master('mkdir ~/.aws')

cluster_ssh.copy_from_local_to_master('/Users/jbsnyder/.aws/jbsnyder.pem', '/home/ubuntu/.aws/jbsnyder.pem')

# create the hosts file for horovod
with open("hosts", "w") as outfile:
    outfile.writelines("localhost slots={}\n".format(GPUS_PER_INSTANCE))
    for worker in range(len(cluster.ips['worker_public_ips'])):
        outfile.writelines("server{} slots={}\n".format(worker, GPUS_PER_INSTANCE))

cluster_ssh.run_on_master('git clone -b nvidia https://github.com/johnbensnyder/resnet/')

cluster_ssh.copy_from_local_to_master('/Users/jbsnyder/PycharmProjects/resnet/hosts', '/home/ubuntu/resnet/hosts')

cluster_ssh.run_on_master('mkdir /home/ubuntu/model')
cluster_ssh.run_on_master('mkdir /home/ubuntu/results')
cluster_ssh.run_on_master('mkdir /home/ubuntu/summaries')
cluster_ssh.run_on_master('mkdir /home/ubuntu/data')

cluster_ssh.run_on_master('./train.sh 32 --data_dir ~/data/tf-imagenet --batch_size 256 --num_epochs 10 --log_dir ~/imagenet_resnet --eval_interval 2')




# ['3.85.209.87', '34.239.135.135', '35.175.209.244']


a_node = infra.EC2Node('jbsnyder-test-node', 'us-east-1')

params = {'az': 'us-east-1b', 'vpc_id': 'vpc-f6570b8d', 'subnet_id': 'subnet-58b35b04',
          'ami_id': 'ami-0757fc5a639fe7666', 'ebs_snapshot_id': 'snap-0876a940246970049',
          'volume_size_gb': 75, 'volume_type': 'gp2', 'key_name': 'jbsnyder',
          'security_group_ids': ['sg-0d5ee02d5f0acb5f3'], 'iam_ec2_role_name': 'jbsnyder',
          'instance_type': 'p3.2xlarge'}

status = a_node.launch(**params)

volume = a_node.ec2_resource.create_volume(AvailabilityZone=params['az'], Size=500, VolumeType='gp2',
                                           TagSpecifications=[{'ResourceType': 'volume',
                                                              'Tags': [{'Key': 'Name',
                                                                        'Value': 'jbsnyder_imagenet'}]}])

instance = a_node.ec2_resource.Instance(a_node.instance_id)

response = instance.attach_volume(Device='/dev/sdh', VolumeId=volume.id)

node_ssh = control.ClusterShell('ubuntu', a_node.public_ip, [], '~/.aws/jbsnyder.pem')

node_ssh.run_on_all('sudo mkfs -t ext4 /dev/xvdh')

node_ssh.run_on_all('sudo mkdir ~/data')

node_ssh.run_on_all('sudo mount /dev/xvdh ~/data')

node_ssh.run_on_all('sudo chmod 777 ~/data')

node_ssh.copy_from_local_to_all('resnet/imagenet_download.py', '/home/ubuntu/imagenet_download.py')

node_ssh.run_on_all("~/anaconda3/envs/tensorflow_p36/bin/python /home/ubuntu/imagenet_download.py")

node_ssh.run_on_all("sudo umount /dev/xvdh")

client = boto3.client('ec2')

response = client.detach_volume(InstanceId=instance.id, VolumeId=volume.id)

client.delete_volume(VolumeId=volume.id)

a_node.terminate()





