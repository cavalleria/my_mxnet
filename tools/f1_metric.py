import mxnet as mx


def micro_f1_score(label, pred):
    from sklearn.metrics import f1_score
    from sklearn.metrics import hamming_loss
    import numpy as np
    pred = np.squeeze(pred)
    pred = np.argsort(pred, axis=1)[:, -1]
    #     result = f1_score(label, pred,average='micro',labels = np.unique(pred))
    result = f1_score(label, pred, average='micro')
    return result


def macro_f1_score(label, pred):
    from sklearn.metrics import f1_score
    from sklearn.metrics import hamming_loss
    import numpy as np
    pred = np.squeeze(pred)
    pred = np.argsort(pred, axis=1)[:, -1]
    result = f1_score(label, pred, average='macro', labels=np.unique(pred))
    #     result = f1_score(label, pred,average='macro')
    return result


def f1_score(type = 'macro'):
    if type == 'macro':
        return mx.metric.create(macro_f1_score)
    else:
        return mx.metric.create(micro_f1_score)
