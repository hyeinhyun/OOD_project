import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

def evaluate(labels, scores, metric='roc',save_to=''):

    if metric == 'roc':
        rocs=roc(labels, scores)
        return rocs

    elif metric == 'best f1':
        return best_f1(labels, scores)
    elif metric == 'test':
        X1 = scores[labels == 0]#ID
        Y1 = scores[labels == 1]#OOD

        fpr95, detect_error, auprin, auprout, auroc=calMetric(X1,Y1,save_to)

        return auroc,fpr95, detect_error, auprin, auprout
    else:
        raise NotImplementedError("Check the evaluation metric.")

#
def best_f1(labels, scores):
    """ 
    Evaluate the best F1 score

    Returns:
        best, acc, sens, spec: the best F1 score, accuracy, sensitivity, specificity
    """
    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    best = 0.0
    best_threshold = -1

    for threshold in thresholds[1:]:
        tmp_scores = scores.clone().detach()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores <  threshold] = 0
        f1 = f1_score(labels, tmp_scores)
        if best < f1:
            best = f1
            best_threshold = threshold
    
    preds =  scores.clone().detach()
    preds[preds >= best_threshold] = 1
    preds[preds <  best_threshold] = 0

    TP = preds[labels == 1].sum().item() # True positive
    CP = (labels == 1).sum().item() # Condition positive = TP + FN
    TN = (labels == 0).sum().item() - preds[labels == 0].sum().item()
    CN = (labels == 0).sum().item()

    acc = (TP + TN) / (CP + CN)
    sens = TP / CP
    spec = TN / CN

    return best, acc, sens, spec, best_threshold

#
def roc(labels, scores):
    """ 
    Evaluate ROC

    Returns:
        auc, eer: Area under the curve, Equal Error Rate
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def calMetric(X1, Y1,save_to=''):
    """
    Args:
         X1 (list): Prediction values for inliers
         Y1 (list): Prediction values for outliers
    """

    thre = 1.0
    tp = sum(np.array(X1) <= thre)
    fn = sum(np.array(X1) > thre)
    tn = sum(np.array(Y1) > thre)
    fp = sum(np.array(Y1) <= thre)


    print('\ntp: %d, fn: %d , tn: %d, fp: %d\n' % (tp, fn, tn, fp))


    X1=X1.tolist()
    Y1=Y1.tolist()
    total_li=X1+Y1
    total_li.sort()
    total_li=np.array(total_li)
    
    ##################################################################
    # auroc
    ##################################################################   
    auroc = 0.0
    fprTemp = 0.0
    for delta in total_li:
        tpr = np.sum(np.sum(X1 <= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 <= delta)) / np.float(len(Y1))
        auroc += (fpr-fprTemp)*tpr
        fprTemp = fpr
    auroc += (1-fpr) * tpr
    
    if save_to:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(save_to)
        plt.close()
    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    fpr=0.0
    total=0
    for delta in total_li:
        tpr = np.sum(np.less_equal(X1, delta)) / np.float(len(X1))
        error2 = np.sum(np.less_equal(Y1, delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
        #print("tpr: %.3f" % tpr)

    fpr95 = fpr/total

    #print("fpr95: %.3f" % fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    detect_error = 1.0
    for delta in total_li:
        fnr = np.sum(np.greater_equal(X1, delta)) / np.float(len(X1))
        fpr = np.sum(np.less_equal(Y1, delta)) / np.float(len(Y1))
        detect_error = np.minimum(detect_error, (fnr + fpr) / 2.0)

    #print("Detection error: %.3f " % detect_error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 0.0
    for delta in total_li:
        tp = np.sum(np.less_equal(X1, delta))
        fp = np.sum(np.less_equal(Y1, delta))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recall - recallTemp) * precision
        recallTemp = recall
    auprin += (1-recall) * precision

    #print("auprin: %.3f" % auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    X1_minus = [-x for x in X1]
    Y1_minus = [-x for x in Y1]
    total_minus=X1_minus+Y1_minus
    total_minus.sort()
    total_minus=np.array(total_minus)
    auprout = 0.0
    recallTemp = 0.0
    for delta in total_minus:
        tp = np.sum(np.less_equal(Y1_minus, delta))
        fp = np.sum(np.less_equal(X1_minus, delta))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1_minus))
        auprout += (recall - recallTemp) * precision
        recallTemp = recall
    auprout += (1-recall) * precision

    #print("auprout: %.3f" % auprout)

    return fpr95, detect_error, auprin, auprout, auroc  
