import mxnet as mx
import os
from config import rec_path,train_rec,train_idx,val_rec,val_idx,batch_size


def get_iterators(batch_size = batch_size, data_shape=(3, 299, 299),shuffle=True):
    train_data = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(rec_path, train_rec),
        path_imgidx = os.path.join(rec_path, train_idx),
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
        path_imgrec=os.path.join(rec_path, val_rec),
        path_imgidx=os.path.join(rec_path, val_idx),
        data_shape=data_shape,
        batch_size=batch_size,
        data_name='data',
        label_name='softmax_label',
        resize = 299,
    )
    return (train_data, val_data)


if __name__ =='__main__':
    (train, val) = get_iterators()
    print(train.next)