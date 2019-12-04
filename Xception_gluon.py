import mxnet as mx
from gluoncv.model_zoo import get_model
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import utils as gutils
import time
from tools.loss import *
import os

from mxboard import *
# sw = SummaryWriter(logdir='./logs')

import logging
logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%Y-%d-%m %H:%M:%S", level=logging.DEBUG)

from gluoncv.loss import FocalLoss


GPU_COUNT = 2
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

per_gup_batch_size = 8
batch_size = GPU_COUNT * per_gup_batch_size
train_epoch = 10
classes = 25

# save_name = 'xception'

# save_name = 'xception_Sigmoid'

# save_name = 'xception_ArcLoss'


save_name = 'xception_FocalLoss_adam'


def get_fine_tune_model(pretrained=None):
    if pretrained == None:
        fine_tune_net = get_model('xception', pretrained=True)
    else:
        fine_tune_net = get_model('xception', pretrained=False)
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(fine_tune_net.conv1)
        net.add(fine_tune_net.bn1)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.conv2)
        net.add(fine_tune_net.bn2)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.block1)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.block2)
        net.add(fine_tune_net.block3)
        net.add(fine_tune_net.midflow)
        net.add(fine_tune_net.block20)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.conv3)
        net.add(fine_tune_net.bn3)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.conv4)
        net.add(fine_tune_net.bn4)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.conv5)
        net.add(fine_tune_net.bn5)
        net.add(fine_tune_net.relu)
        net.add(fine_tune_net.avgpool)
        net.add(fine_tune_net.flat)
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(25))
        if pretrained == 'best':
            net.load_parameters('weights/best_' + save_name + '.params')
            # net.load_parameters('weights/gluon_xception_CosLoss-0005.params')
            # net.imports('eights/gluon_xception_CosLoss-0005.params')
        elif pretrained == 'last':
            net.load_parameters('weights/last_' + save_name + '.params')
        elif pretrained == None :
            net[-1].collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
        else:
            net.load_parameters('weights/' + pretrained)
    net.hybridize()
    net.collect_params().reset_ctx(ctx=ctx)
    return net


# loss = ArcLoss(classes = classes,m = 0.5,s = 64)-

loss = FocalLoss(num_class = classes)

# loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
#获取微调网络
net = get_fine_tune_model()

#构建数据集
train_transform= transforms.Compose([
                      transforms.Resize(299),
                      transforms.RandomSaturation(0.2),
                      transforms.RandomContrast(0.2),
                      transforms.RandomFlipLeftRight(),
                      transforms.RandomFlipTopBottom(),
                      transforms.ToTensor(),
                                  ])

val_transform= transforms.Compose([
                      transforms.Resize(299),
                      # transforms.RandomSaturation(0.2),
                      # transforms.RandomContrast(0.2),
                      # transforms.RandomFlipLeftRight(),
                      # transforms.RandomFlipTopBottom(),
                      transforms.ToTensor(),
                                  ])

train_data = gluon.data.vision.ImageRecordDataset('img_file/train.rec').transform_first(train_transform)
train_loader = gluon.data.DataLoader(train_data, batch_size=batch_size,num_workers = 4,shuffle = True)
val_data = gluon.data.vision.ImageRecordDataset('img_file/val.rec').transform_first(val_transform)
val_loader = gluon.data.DataLoader(val_data, batch_size=batch_size,num_workers = 4)

file_name = 'label_name.txt'
with open(file_name, 'r') as fileopen:
    name_list = [line.strip() for line in fileopen]

def accuracy(output,label):
    #计算准确率
    acc = mx.metric.Accuracy()
    predictions = mx.nd.argmax(output, axis=1)
    acc.update(preds=predictions, labels=label)
    # print (mx.ndarray.Activation(data = output,act_type = 'sigmoid'))
    # print (label)
    # acc = mx.metric.TopKAccuracy(3)
    # acc.update(label, mx.ndarray.SoftmaxActivation(data = output))
    # acc.update(label, mx.ndarray.Activation(data = output,act_type = 'sigmoid'))

    return acc.get()[1]


def evaluate_accuracy(data_iterator, net, ctx=ctx):
    acc = 0.
    for i,(X, y) in enumerate(data_iterator):
        X = X / 255
        data = gutils.split_and_load(X, ctx, even_split=False)
        label = gutils.split_and_load(y, ctx, even_split=False)
        output = [(net(data), label) for data, label in zip(data, label)]
        for result in output:
            acc += accuracy(result[0],result[1])
        # if i + 1 >= 30 :
        #     return acc / (30 * len(ctx))
    return acc / (len(data_iterator) * len(ctx))


def ConfusionMatrix(output,label):
    #计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    predictions = mx.nd.argmax(output, axis=1)
    cm=confusion_matrix(label.asnumpy(), predictions.asnumpy(),labels = range(25))
    return cm


def evaluate_confusion_matrix(data_iterator,net,ctx = ctx):
    import numpy as np
    cm = np.zeros([classes,classes])
    for i,(X, y) in enumerate(data_iterator):
        X = X / 255
        data = gutils.split_and_load(X, ctx, even_split=False)
        label = gutils.split_and_load(y, ctx, even_split=False)
        output = [(net(data), label) for data, label in zip(data, label)]
        for result in output:
            cm += ConfusionMatrix(result[0],result[1])
    return cm


def train(num_gpus = GPU_COUNT, batch_size = batch_size, lr = 0.001 , train_epoch = train_epoch):
    #训练模型
    train_iter, test_iter = train_loader , val_loader
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    # print('running on:', ctx)
    logging.info('running on:GPU')
    # loss = gluon.loss.SoftmaxCrossEntropyLoss()

    best_acc = 0.
    # loss = L2Softmax(classes = 25,alpha = 10)
    trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': lr})

    for epoch in range(train_epoch):
        train_loss = 0.
        start = time.time()
        for i,(X, y) in enumerate(train_iter):
            X = X / 255
            gpu_Xs = gutils.split_and_load(X, ctx,even_split = False)
            gpu_ys = gutils.split_and_load(y, ctx,even_split = False)
            with autograd.record():
                # ls = [loss(net(gpu_X), mx.nd.one_hot(gpu_y,classes))
                #       for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
                ls = [loss(net(gpu_X), gpu_y)
                      for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            ls_list = [nd.mean(i).asscalar() for i in ls]
            train_loss += sum(ls_list) / len(ls_list)
            if (i + 1) % 50 == 0:
                #打印训练日志
                logging.info("epoch[{epoch}]  batch_num[{batch_num}]  epochtrain_loss : {loss}".format(
                    epoch = epoch + 1,batch_num = i +1 ,loss = train_loss/(i +1)))
                # test_acc = evaluate_accuracy(test_iter, net, ctx)
                # train_time = time.time() - start
                # logging.info('epoch %d, time %.1f sec, test acc %.7f' % (
                #     epoch + 1, train_time, test_acc))
                # net.save_parameters('weights/best_' + save_name + '.params')
        nd.waitall()
        train_time = time.time() - start
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        logging.info('epoch[%d], time %.5f sec, test acc %.7f' % (
            epoch + 1, train_time, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            net.save_parameters('weights/best_' + save_name + '.params')
        if ( epoch + 1 ) % 5 == 0 :
            net.save_parameters('weights/gluon_' + save_name + str(epoch + 1) + '.params')
        net.save_parameters('weights/gluon_' + save_name + str(epoch + 1) + '.params')
        net.save_parameters('weights/last_' + save_name + '.params')


def eval(num_gpus = GPU_COUNT):
    #测试模型准确率
    train_iter, test_iter = train_loader , val_loader
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    logging.info('running on:GPU')

    test_acc = evaluate_accuracy(test_iter, net, ctx)
    logging.info('test acc %.7f' % ( test_acc ))



def save_comfusion_metrix(num_gpus = GPU_COUNT,filename = save_name):
    #保存混淆矩阵
    import pandas as pd
    train_iter, test_iter = train_loader, val_loader
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    logging.info('running on:GPU')
    test_cm = evaluate_confusion_matrix(test_iter, net, ctx)
    cm = pd.DataFrame(test_cm)
    cm.to_excel(filename + '_cm.xlsx')

def eval_sym_model(model_json,model_params,name):
    #保存symbol模型生成混淆矩阵
    ctx = [mx.gpu(i) for i in range(2)]
    net = gluon.nn.SymbolBlock.imports(model_json, ['data'],model_params, ctx=ctx)
    import pandas as pd
    train_iter, test_iter = train_loader, val_loader
    logging.info('running on:GPU')
    test_cm = evaluate_confusion_matrix(test_iter, net, ctx)
    cm = pd.DataFrame(test_cm)
    cm.to_excel(name + '_cm.xlsx')

def predict():
    import numpy as np

    from scipy import misc

    # predict_data = gluon.data.vision.datasets.ImageFolderDataset('predict_img').transform_first(val_transform)
    # predict_iterator = gluon.data.DataLoader(predict_data, batch_size=batch_size, num_workers=4)

    for item,(X, y) in enumerate(train_loader):
        logging.info(len(train_loader))
        logging.info (item)
        X = X / 255
        data = gutils.split_and_load(X, ctx, even_split=False)
        label = gutils.split_and_load(y, ctx, even_split=False)
        output = [mx.ndarray.SoftmaxActivation(data=net(i)) for i in data]
        for i in range(len(output)):
            y_pred = output[i].asnumpy().tolist()
            y_pred_index = np.argmax(y_pred, axis=1)
            y_pred_prob = np.max(y_pred,axis = 1)
            y_label = label[i].asnumpy().tolist()
            y_label = np.array(y_label)
            error_index = [i for i in range(len(y_pred_index)) if y_pred_index[i] != y_label[i]]
            if len(error_index):
                for j in error_index:
                    predict_name = name_list[int(y_pred_index.tolist()[j])]
                    label_name = name_list[int(y_label.tolist()[j])]
                    predict_prob = str(y_pred_prob.tolist()[j])
                    img_array = data[i][j] * 255
                    misc.imsave('img_file/hard_train/' + label_name + '/' +
                                # str(item) + '_' + str(i) + '_' + str(j) + '_' +
                                predict_prob + '-' +
                                label_name + '-->' + predict_name + '.bmp',
                                img_array.asnumpy().transpose(1,2,0))
    return 0


def save_hard_train_img():
    import numpy as np




    # predict_data = gluon.data.vision.datasets.ImageFolderDataset('/media/lin/disk_8T/sheldon/Train18_1p1/train').transform_first(val_transform)
    # predict_iterator = gluon.data.DataLoader(predict_data, batch_size=batch_size, num_workers=4)
    # y_pred = []
    # for i,(X, y) in enumerate(predict_iterator):
    #     X = X / 255
    #     data = gutils.split_and_load(X, ctx, even_split=False)
    #     output = [mx.ndarray.SoftmaxActivation(data=net(i)) for i in data]
    #     # output = [mx.ndarray.Activation(data = net(i),act_type='sigmoid') for i in data]
    #     print(output)
    #     for result in output:
    #         p = result.asnumpy().tolist()
    #         y_pred.extend(p)
    #
    #     # if i + 1 >= 10 :
    #     #     return acc / (10 * len(ctx))
    # y_pred_index = np.argmax(y_pred,axis = 1)
    # return y_pred_index


    from PIL import Image
    im = Image.fromarray(A)
    im.save("your_file.jpeg")

if __name__ == "__main__":

    #读取json与params生成混淆矩阵
    # model_json = '/home/lin/sheldon/my_mxnet/weights/gluon_xception-symbol.json'
    # model_params = '/home/lin/sheldon/my_mxnet/weights/gluon_xception-0005.params'
    # name = 'xception_softmax_05'
    # eval_sym_model(model_json,model_params,name)


    train()
    # save_comfusion_metrix()
    # print(eval())
    # print (predict())
    # save_hard_train_img()