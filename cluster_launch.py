INSTANCE_TYPE = "p3dn.24xlarge"
GPUS_PER_INSTANCE = 8
INSTANCE_COUNT = 1
CLUSTER_NAME = "nvidia_docker_large"

import os
from time import sleep
import yaml
from ec2_cluster import infra, control

'''
Read configuration information
'''

with open('/Users/jbsnyder/PycharmProjects/resnet/user_info.cred') as infile:
    user_info = yaml.load(infile, Loader=yaml.Loader)

with open('/Users/jbsnyder/PycharmProjects/resnet/cluster_config.yaml') as infile:
    cluster_info = yaml.load(infile, Loader=yaml.Loader)

'''
Create cluster instance
'''

cluster = infra.EC2NodeCluster(INSTANCE_COUNT,
                               "{}-{}-".format(user_info['USER_NAME'],
                                               CLUSTER_NAME), 'us-east-1')

'''
Launch cluster
'''

params = {'az': cluster_info['AVAILABILITY_ZONE'],
          'vpc_id': cluster_info['VPC_ID'],
          'subnet_id': cluster_info['SUBNET_ID'],
          'ami_id': cluster_info['AMI_ID'],
          'ebs_snapshot_id': cluster_info['EBS_SNAPSHOT_ID'],
          'volume_gbs': cluster_info['VOLUME_GBS'],
          'volume_type': cluster_info['VOLUME_TYPE'],
          #'key_name': key_pair['KeyName'],
          'key_name': user_info['KEY_PAIR'],
          'security_group_ids': cluster_info['SECURITY_GROUP_IDS'],
          'iam_ec2_role_name': user_info['IAM_EC2_ROLE'],
          'instance_type': INSTANCE_TYPE}

status = cluster.launch(**params)

'''
Need to give it a minute to start
'''

sleep(60)

'''
Create ssh connection
'''

cluster_ssh = control.ClusterShell('ubuntu', cluster.public_ips[0], cluster.public_ips[1:], user_info['KEY_FILE'])

'''
If not using a p3dn, create and ebs volume
in either case, mount for use
'''

def mount_volumes(device='/dev/xvdh', location='~/model'):
    cluster_ssh.run_on_all('sudo mkfs -t ext4 {}'.format(device))
    cluster_ssh.run_on_all('sudo mkdir {}'.format(location))
    cluster_ssh.run_on_all('sudo mount {} {}'.format(device, location))
    cluster_ssh.run_on_all('sudo chmod 777 {}'.format(location))

def attach_ebs(instance, ebs_size=500, device='/dev/sdh'):
    tags = [{'ResourceType': 'volume',
             'Tags': [{'Key':'Name',
                       'Value':'{}-{}-ebs'.format(user_info['USER_NAME'], instance.instance_id)}]}]
    a_volume = cluster.ec2_client.create_volume(AvailabilityZone=cluster_info['AVAILABILITY_ZONE'],
                                                Size=ebs_size,
                                                VolumeType='gp2',
                                                TagSpecifications=tags)
    # wait for volume to become available
    while cluster.ec2_client.describe_volumes(VolumeIds=[a_volume['VolumeId']])['Volumes'][0]['State'] != 'available':
        sleep(2)

    instance.attach_volume(Device=device, VolumeId=a_volume['VolumeId'])
    return instance.instance_id, a_volume

def attach_ebs_cluster(cluster, ebs_size=500, device='/dev/sdh'):
    instances = [cluster.ec2_resource.Instance(i) for i in cluster.instance_ids]
    volumes = [attach_ebs(instance) for instance in instances]
    return volumes

if INSTANCE_TYPE != 'p3dn.24xlarge':
    volumes = attach_ebs_cluster(cluster)
    mount_volumes()
else:
    mount_volumes(device='/dev/nvme2n1')

'''
Install AWS CLI since using Nvidia IAM
'''

cluster_ssh.run_on_all('sudo apt install -y awscli')
cluster_ssh.run_on_all('mkdir ~/.aws')
cluster_ssh.copy_from_local_to_all('/Users/{}/.aws/config'.format(user_info['USER_NAME']), '/home/ubuntu/.aws/config')
cluster_ssh.copy_from_local_to_all('/Users/{}/.aws/credentials'.format(user_info['USER_NAME']), '/home/ubuntu/.aws/credentials')

cluster_ssh.copy_from_local_to_master(user_info['KEY_FILE'], '/home/ubuntu/.ssh/{}'.format(user_info['KEY_PAIR']+'.pem'))

'''
Create config file of worker nodes
'''

def worker_config(host, ip_address, port=22, user='ubuntu', identity_file=user_info['KEY_PAIR']+'.pem'):
    # format ip address
    address = 'ec2-{}.compute-1.amazonaws.com'.format('-'.join(ip_address.split('.')))
    formatted_string = """Host {}
    HostName {}
    port {}
    user {}
    IdentityFile ~/.ssh/{}
    IdentitiesOnly yes

""".format(host, address, port, user, identity_file)
    return formatted_string

with open("worker_config", "w") as outfile:
    for num, address in enumerate(cluster.ips['worker_public_ips']):
        outfile.writelines(worker_config('server{}'.format(num), address))

cluster_ssh.copy_from_local_to_master(os.path.join(os.getcwd(), 'worker_config'), '/home/ubuntu/.ssh/config')

# create the hosts file for horovod
with open("hosts", "w") as outfile:
    outfile.writelines("localhost slots={}\n".format(GPUS_PER_INSTANCE))
    for worker in range(len(cluster.ips['worker_public_ips'])):
        outfile.writelines("server{} slots={}\n".format(worker, GPUS_PER_INSTANCE))

cluster_ssh.copy_from_local_to_master(os.path.join(os.getcwd(), 'hosts'), '/home/ubuntu/hosts')

cluster_ssh.run_on_all('mkdir /home/ubuntu/model/data')
cluster_ssh.run_on_all('mkdir /home/ubuntu/model/model')
cluster_ssh.run_on_all('mkdir /home/ubuntu/model/results')
cluster_ssh.run_on_all('mkdir /home/ubuntu/model/summaries')

path='/home/ubuntu/model/data/tf-imagenet/'

print(cluster.ips)

'''
Download imagenet data
'''

cluster_ssh = control.ClusterShell('ubuntu', cluster.public_ips[0], cluster.public_ips[1:], user_info['KEY_FILE'])
cluster_ssh.run_on_all(
    'aws s3 cp s3://aws-tensorflow-benchmarking/imagenet-armand/validation-480px/ {} --recursive --quiet'.format(path))
cluster_ssh.run_on_all(
    'aws s3 cp s3://aws-tensorflow-benchmarking/imagenet-armand/train-480px/ {} --recursive --quiet'.format(path))

'''
Pull nvidia docker
'''

cluster_ssh.run_on_all('docker pull nvcr.io/nvidia/tensorflow:19.06-py3')

'''
clone model repo
'''

cluster_ssh.run_on_master('cd /home/ubuntu/model/ && \
                           git clone https://{}:{}@github.com/johnbensnyder/resnet/'.format(user_info['GITHUB_ID'],
                                                                                          user_info['GIT_PASS']))

'''
docker run -it --rm -v /home/ubuntu/data:/workspace/model/ \
                        nvcr.io/nvidia/tensorflow:19.06-py3
'''

'''
python /workspace/nvidia-examples/resnet50v1.5/main.py \
--data_dir=/workspace/model/data/tf-imagenet/ \
--num_iter=100 --iter_unit=epoch \
--results_dir=/workspace/data/results/ \
--batch_size=256 \
--use_tf_amp --use_xla
'''

'''
python /workspace/model/model/resnet/RN50v1.5/main.py \
--data_dir=/workspace/model/tf-imagenet/ \
--num_iter=100 --iter_unit=epoch \
--results_dir=/workspace/data/results/ \
--batch_size=256 \
--use_tf_amp --use_xla
'''