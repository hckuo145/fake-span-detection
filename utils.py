import numpy as np
import sklearn.metrics

    
def compute_acc(true, pred):
    return np.mean([ t == round(p) for t, p in zip(true, pred) ]) 


def compute_eer(true, pred, pos_label=1):
    # all fpr, tpr, fnr, tnr, threshold are lists (in the format of np.array)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true, pred, pos_label=pos_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

    # eer from fnr and eer from fpr should be identical but they can be differ in reality
    eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_fnr + eer_fpr) / 2
    
    return eer