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
from tools.f1_metric import f1_score



model_name = 'xception'
root_dir = '/home/lin/sheldon/my_mxnet'
# weights_name = 'gluoncv-xception'
num_classes = 25
num_gpus = 2
batch_per_gpu = 8
train_epoch = 10


batch_size = batch_per_gpu * num_gpus
rec_path = os.path.join(root_dir, 'img_file')


predict_path = os.path.join(root_dir, 'predict_img')


prefix = os.path.join(root_dir, 'weights/xception')
prefix_epoch = 10


def get_iterators(batch_size = batch_size, data_shape=(3, 299, 299),shuffle=True,root_path = rec_path):
    train_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(root_path, 'train.rec'),
        path_imgidx=os.path.join(root_path, 'train.idx'),
        data_shape=data_shape,
        batch_size=batch_size,
        resize = 299,
        saturation = 0.2,
        contrast = 0.2,
        shuffle=shuffle,
        rand_mirror = True,
        brightness = 0.2,
        data_name='data',
        label_name='softmax_label',
        rotate = 180
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(root_path, 'val.rec'),
        path_imgidx=os.path.join(root_path, 'val.idx'),
        data_shape=data_shape,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        resize = 299,
    )
    return (train_data, val_data)


def get_eval_metrics(metrics_list):
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics_1 = mx.metric.Accuracy()
    eval_metrics_2 = mx.metric.TopKAccuracy(5)
    metrics_list.append([eval_metrics_1,eval_metrics_2])
    for child_metric in metrics_list:
        eval_metrics.add(child_metric)
    # (train, val) = get_iterators(batch_size)
    # a = mod.score(eval_data = val,eval_metric = eval_metrics)
    # a = dict(a)
    return eval_metrics

def predict(data,prefix = prefix,prefix_epoch = prefix_epoch,num_batch = None):
    mod = get_model(prefix=prefix, prefix_epoch=prefix_epoch,batch_size = 4)
    pred_result = mod.predict(data, num_batch = num_batch).asnumpy()
    pred_result = np.squeeze(pred_result)
    result = np.argsort(pred_result, axis=1)[:, -1]
    return result
    # return pred_result


def val_model():
    eval_metrics = get_eval_metrics([f1_score()])
    mod = get_model()
    (train, val) = get_iterators(batch_size)
    eval_result = dict(mod.score(eval_data=val, eval_metric=eval_metrics,num_batch = 1))
    return eval_result


def get_model(prefix = prefix,prefix_epoch = prefix_epoch,batch_size=batch_size):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, prefix_epoch)
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=sym, context=devs,
                        data_names=["data"], label_names=None)
    mod.bind(for_training=False, data_shapes=[("data", (batch_size, 3, 299, 299))],
             label_shapes=[("softmax_label", (batch_size,))])
    # 设定模型参数
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

if __name__ == '__main__':
    predict_data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(predict_path, 'predict.rec'),
        path_imgidx=os.path.join(predict_path, 'predict.idx'),
        data_shape=(3,299,299),
        data_name='data',
        label_name='softmax_label',
        batch_size=1,
        resize=299,
    )
    print (predict(data = predict_data))