import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

def get_apcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest bpcer at which the probability of apcer == op
    @rtype: float
    """
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]

def get_bpcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest apcer at which the probability of bpcer == op
    @rtype: float
    """
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]

def performances_compute(prediction_scores, gt_labels, threshold_type='eer', op_val=0.1, verbose=True):
    # fpr = apcer, 1-tpr = bpcer
    # op_val: 0 - 1
    # gt_labels: list of ints,  0 for attack, 1 for bonafide
    # prediction_scores: list of floats, higher value should be bonafide
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1)
    bpcer = 1 - tpr
    val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
    val_auc = auc(fpr, tpr)

    if threshold_type=='eer':
        threshold = eer_threshold
    elif threshold_type=='apcer':
        _, _, threshold = get_apcer_op(fpr, bpcer, threshold, op_val)
    elif threshold_type=='bpcer':
        _, _, threshold = get_bpcer_op(fpr, bpcer, threshold, op_val)
    else:
        threshold = 0.5

    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    threshold_APCER = type2 / num_fake
    threshold_BPCER = type1 / num_real
    threshold_ACER = (threshold_APCER + threshold_BPCER) / 2.0

    if verbose is True:
        print(f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')

    return val_auc, val_eer, [threshold, threshold_APCER, threshold_BPCER, threshold_ACER]

def evalute_threshold_based(prediction_scores, gt_labels, threshold):
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER
