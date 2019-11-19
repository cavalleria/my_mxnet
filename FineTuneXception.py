import mxnet as mx
import os, time, shutil
import logging
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
import numpy as np
import gluoncv
from gluoncv.utils import export_block,viz,makedirs




model_name = 'xception'
root_dir = '/home/lin/sheldon/my_mxnet'
weights_name = 'weights/gluoncv-xception'

num_gpus = 2
batch_per_gpu = 8
train_epoch = 10



batch_size = batch_per_gpu * num_gpus
rec_path = os.path.join(root_dir, 'img_file')
prefix = os.path.join(root_dir, 'weights/xception')



class FineTuneXception(object):

    def __init__(self,args):
        self.model_name = 'xception'
        self.num_classes = 25
        self.num_gpus = args.num_gpus
        self.batch_per_gpu = args.batch_per_gpu
        self.train_epoch = args.train_epoch




        self.get_weights()

    def get_weights(model_name = self.model_name):
        """
        :param model_name:
        :return: None

        """
        finetune_net = get_model(model_name, pretrained=True)
        export_block('weights/gluoncv-'+ model_name, finetune_net, preprocess=None, layout='CHW')


    def get_fine_tune_model(symbol,arg_params,num_classes,layer_name = 'flatten0'):
        """
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
            lr = 0.01):
        """
        :param symbol: the network symbol
        :param arg_params: the argument parameters of the network model
        :param aux_params: the aux_params parameters of the network model
        :return:

        """

        checkpoint = mx.callback.do_checkpoint(prefix = prefix,period = 5)

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
            num_epoch = train_epoch,
            arg_params = arg_params,
            aux_params = aux_params,
            allow_missing = True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 10),
            kvstore = 'device',
            optimizer = 'Adagrad',
            optimizer_params = {'learning_rate':lr,},#学习率有点大后期准确率波动大。
            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric = ['ce','acc','f1'],
            epoch_end_callback = checkpoint)
        metric_acc = mx.metric.Accuracy()
        return mod.score(val, metric_acc)


    def show_img(self,img_iterator):
        import matplotlib.pyplot as plt
        # img_batch_data.reset()
        data_batch = img_iterator.next()
        data = data_batch.data[0]
        plt.figure()
        for i in range(8):
            _image = data[i].astype('uint8').asnumpy().transpose((1, 2, 0))
            plt.subplot(2, 4, i + 1)
            plt.imshow(_image)
        plt.show()


    def train(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint(weights_name, 0)
        (new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
        (train, val) = get_iterators(batch_size)
        mod_score = fit_model(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
        # assert mod_score > 0.7, "Low training accuracy."
        print ("model val accurace is %.4f"%mod_score)

    def which_train():
        if "train" :
            pass
        else:
            pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model', type=str, default='model/resnet-18')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--begin-epoch', type=int, default=0)
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--resize-train', type=int, default=256)
    parser.add_argument('--resize-val', type=int, default=224)
    parser.add_argument('--data-train-rec', type=str, default='data/data_train.rec')
    parser.add_argument('--data-train-idx', type=str, default='data/data_train.idx')
    parser.add_argument('--data-val-rec', type=str, default='data/data_val.rec')
    parser.add_argument('--data-val-idx', type=str, default='data/data_val.idx')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--save-result', type=str, default='output/resnet-18/')
    parser.add_argument('--num-examples', type=int, default=22500)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name', type=str, default='resnet-18')
    parser.add_argument('--random-mirror', type=int, default=1,
                        help='if or not randomly flip horizontally')
    parser.add_argument('--max-random-contrast', type=float, default=0.3,
                        help='Chanege the contrast with a value randomly chosen from [-max, max]')
    parser.add_argument('--max-rotate-angle', type=int, default=15,
                        help='Rotate by a random degree in [-v,v]')
    parser.add_argument('--layer-name', type=str, default='flatten0',
                        help='the layer name before fullyconnected layer')
    parser.add_argument('--factor', type=float, default=0.2, help='factor for learning rate decay')
    parser.add_argument('--step', type=int, default=5, help='step for learning rate decay')
    parser.add_argument('--from-scratch', type=bool, default=False,
                        help='Whether train from scratch')
    parser.add_argument('--fix-pretrain-param', type=bool, default=False,
                        help='Whether fix parameters of pretrain model')
    args = parser.parse_args()
    return args

def get_iterators(batch_size = batch_size, data_shape=(3, 299, 299),shuffle=True):
        train_data = mx.io.ImageRecordIter(
            path_imgrec = os.path.join(rec_path, 'train.rec'),
            path_imgidx = os.path.join(rec_path, 'train.idx'),
            data_shape = data_shape,
            batch_size = batch_size,
            data_name='data',
            label_name='softmax_label',
            resize = 299,
            saturation = 0.2,
            contrast = 0.2,
            shuffle = shuffle,
            rand_mirror = True,
            brightness = 0.2,
            rotate = 180
        )
        val_data = mx.io.ImageRecordIter(
            path_imgrec=os.path.join(rec_path, 'val.rec'),
            path_imgidx=os.path.join(rec_path, 'val.idx'),
            data_shape=data_shape,
            batch_size=batch_size,
            data_name='data',
            label_name='softmax_label',
            resize = 299,
        )
        return (train_data, val_data)

if __name__ =='__main__':
    # train()
    # (train, val) = get_iterators(batch_size)
    # show_img(train)
    get_weights(model_name)