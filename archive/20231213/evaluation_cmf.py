
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : evaluation_cmf.py
@Time : 2023/11/09 11:19:56
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# use for multilabel-indicator and continuous-multioutput targets
import numpy as np
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]

# small error indicate better performance
def get_coverage_error(targets, scores):
    return coverage_error(targets, scores)

def get_label_ranking_average_precision_score(targets, scores):
    return label_ranking_average_precision_score(targets, scores)

def get_label_ranking_loss(targets, scores):
    return label_ranking_loss(targets, scores)


# use for multilabel-indicator 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def get_f1(targets, scores):
    return f1_score(targets, scores > CUTOFF, average = "samples")

def get_accuracy(targets, scores):
    return accuracy_score(targets, scores > CUTOFF)


# use for singlelabel binary classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

def get_auc(targets, scores):
    return roc_auc_score(targets, scores)

precision, recall, thresholds = precision_recall_curve(targets, inputs)
