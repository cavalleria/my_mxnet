import mxnet as mx
import os
import logging
from gluoncv.model_zoo import get_model
from gluoncv.utils import export_block,viz,makedirs
from config import *
from data_generator import get_iterators
from tools.f1_metric import f1_score


def get_weights(model_name):
    """
    下载预训练参数并保存到weights目录下,以gluoncv开头
    :param model_name:
    :return: None

    """
    finetune_net = get_model(model_name, pretrained=True)
    export_block('weights/gluoncv-'+ model_name, finetune_net, preprocess=None, layout='CHW')


def get_fine_tune_model(symbol,arg_params,num_classes,layer_name = 'flatten0'):
    """
    对later_name层后面的网络进行微调,返回微调后的symbol与arg_params
    :param symbol: the pretrained network symbol
    :param arg_params: the argument parameters of the pretrained model
    :param num_classes: the number of classes for the fine-tune datasets
    :param layer_name: the layer name before the last fully-connected layer
    :return: fine-tune symbol,new_arg_params

    """
    all_layers = symbol.get_internals()
    net = all_layers[[i for i in all_layers if layer_name in i.name][0].name + '_output']
    net = mx.symbol.Dropout(data=net, p=0.5, name='dropout0')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


def fit_model(
        symbol,
        arg_params,
        aux_params,
        train,
        val,
        batch_size = batch_size,
        num_gpus = num_gpus,
        epoch = epoch,
        lr = 0.01,
        checkpoint_period = 5):
    """
    训练模型
    :param symbol: the network symbol
    :param arg_params: the argument parameters of the network model
    :param aux_params: the aux_params parameters of the network model
    :return: 模型对验证集的'ce','acc','f1'

    """

    checkpoint = mx.callback.do_checkpoint(prefix = prefix,period = checkpoint_period)

    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)

    #print and save train logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_headler = logging.StreamHandler()
    logger.addHandler(stream_headler)
    file_hadler = logging.FileHandler('train.log', 'w')
    logger.addHandler(file_hadler)
    logger.info("train logging")

    mod.fit(train, val,
        num_epoch = epoch,
        arg_params = arg_params,
        aux_params = aux_params,
        allow_missing = True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore = 'device',
        optimizer = 'Adagrad',
        optimizer_params = {'learning_rate':lr,},#学习率有点大后期准确率波动大。
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric = ['ce'],
        epoch_end_callback = checkpoint)
    metric_acc = mx.metric.Accuracy()
    return mod.score(val, metric_acc)


# def get_iterators(batch_size = batch_size, data_shape=(3, 299, 299),shuffle=True):
#     train_data = mx.io.ImageRecordIter(
#         path_imgrec = os.path.join(rec_path, 'train.rec'),
#         path_imgidx = os.path.join(rec_path, 'train.idx'),
#         data_shape = data_shape,
#         batch_size = batch_size,
#         data_name='data',
#         label_name='softmax_label',
#         resize = 299,
#         saturation = 0.2,
#         contrast = 0.2,
#         shuffle = shuffle,
#         rand_mirror = True,
#         brightness = 0.2,
#         rotate = 180
#     )
#     val_data = mx.io.ImageRecordIter(
#         path_imgrec=os.path.join(rec_path, 'val.rec'),
#         path_imgidx=os.path.join(rec_path, 'val.idx'),
#         data_shape=data_shape,
#         batch_size=batch_size,
#         data_name='data',
#         label_name='softmax_label',
#         resize = 299,
#     )
#     return (train_data, val_data)
#
# def show_img(img_iterator):
#     import matplotlib.pyplot as plt
#     # img_batch_data.reset()
#     data_batch = img_iterator.next()
#     data = data_batch.data[0]
#     plt.figure()
#     for i in range(8):
#         _image = data[i].astype('uint8').asnumpy().transpose((1, 2, 0))
#         plt.subplot(2, 4, i + 1)
#         plt.imshow(_image)
#     plt.show()


def train():
    """
    :param weights_name: 训练需要加载的模型名称
    :param epoch: 在哪一个epoch的基础上继续训练

    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(weights_name, weights_epoch)
    if weights_name == 'weights/gluoncv-xception':
        #对预训练网络进行微调
        (new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
        print("finished get fine tune model!!!")
    else:
        #读取上一次训练的参数
        new_sym, new_args = sym, arg_params
        print("loaded trained model!!!")
    (train, val) = get_iterators(batch_size)
    mod_score = fit_model(new_sym, new_args, aux_params, train, val, batch_size, num_gpus,epoch)
    print ("model val accurace is %.4f"%mod_score)



if __name__ =='__main__':
    train()
    # (train, val) = get_iterators(batch_size)
    # show_img(train)
    # get_weights(model_name)
    # print (batch_size)