import os

#config
#########################################################
model_name = 'xception'
root_dir = '/home/lin/sheldon/my_mxnet'

# weights_name = 'weights/gluoncv-xception' #开始训练的权重
# weights_epoch = 0 #开始训练读取的epoch数

weights_name = 'weights/xception' #开始训练的权重
weights_epoch = 10 #开始训练读取的epoch数

train_rec = 'train.rec'
train_idx = 'train.idx'
val_rec = 'val.rec'
val_idx = 'val.idx'


num_classes = 25
num_gpus = 2
batch_per_gpu = 8
epoch = 20  #训练迭代次数


batch_size = batch_per_gpu * num_gpus
rec_path = os.path.join(root_dir, 'img_file')
prefix = os.path.join(root_dir, 'weights/xception')
#########################################################