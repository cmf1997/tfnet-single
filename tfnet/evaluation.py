#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : evaluation.py
@Time : 2023/11/09 11:20:10
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve, auc
from tfnet.all_tfs import all_tfs
from logzero import logger
import pdb
import warnings

warnings.filterwarnings('error')  
sns.set_theme(style="ticks")
palette = ['#DC143C', '#4169E1','#ff69b4']

__all__ = ['CUTOFF', 'get_auc', 'get_aupr', 'get_pcc', 'get_srcc', 'get_recall', 'get_f1', 'get_precision', 'get_balanced_accuracy_score','get_label_ranking_average_precision_score', 'get_group_metrics', 'output_eval', 'output_predict']

CUTOFF = 0.8


# code


def get_auc(targets, scores):
    auc = roc_auc_score(targets, scores)
    return auc


def get_recall(targets, scores, cutoff = CUTOFF):
    recall = recall_score(targets, scores > cutoff, zero_division=1.0)
    return recall


def get_precision(targets, scores, cutoff = CUTOFF):
    precision = precision_score(targets, scores > cutoff, zero_division=1.0)
    return precision 


def get_aupr(targets, scores):
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall


def get_f1(targets, scores, cutoff=CUTOFF):
    f1 = f1_score(targets, scores > cutoff, zero_division=1.0)
    return f1


def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]


def get_srcc(targets, scores):
    return spearmanr(targets, scores)[0]


def get_accuracy_score(targets, scores, cutoff=CUTOFF):
    accuracy = accuracy_score(targets, scores> cutoff)
    return accuracy


def get_balanced_accuracy_score(targets, scores, cutoff=CUTOFF):
    accuracy = balanced_accuracy_score(targets, scores> cutoff)
    return accuracy

#try :
#    balanced_accuracy_score(targets[i, :], scores[i, :]> CUTOFF)   # due to all 0 in y_true
#except Warning as e:
#        pdb.set_trace()


def output_eval(chrs, starts, stops, targets_lists, scores_lists, tfs, celltypes, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path = output_path.with_suffix('.eval.tsv')

    metrics = []
    metrics.append(get_auc(targets_lists, scores_lists))
    metrics.append(get_aupr(targets_lists, scores_lists))
    metrics.append(get_pcc(targets_lists, scores_lists))
    metrics.append(get_srcc(targets_lists, scores_lists))
    metrics.append(get_recall(targets_lists, scores_lists))
    metrics.append(get_f1(targets_lists, scores_lists))
    metrics.append(get_accuracy_score(targets_lists, scores_lists))
    metrics.append(get_balanced_accuracy_score(targets_lists, scores_lists))

    # ---------------------- plot ---------------------- #
    eval_data = pd.DataFrame({
                    "AUC" : get_auc(targets_lists, scores_lists),
                    "AUPR": get_aupr(targets_lists, scores_lists),
                    "ACCURACY": get_accuracy_score(targets_lists, scores_lists),
                    "BALANCED_ACCURACY": get_balanced_accuracy_score(targets_lists, scores_lists)
                    }, index=[0])
    sns.barplot(data=eval_data, alpha=0.7)
    plt.tick_params(axis='x', labelrotation=45)
    plt.savefig(output_path.with_suffix('.eval.pdf')) 


    sns.jointplot(x="Target", y="Prediction", data=pd.DataFrame({"Target":targets_lists, "Prediction": scores_lists}),
                    kind="reg", #truncate=False,
                    xlim=(-0.2, 1.2), ylim=(-0.2, 1.2),
                    #color="m", 
                    height=5)
    plt.text(0.3,0.85,f'PCC: {get_pcc(targets_lists, scores_lists):.5f}')
    plt.text(0.3,0.75,f'SRCC: {get_srcc(targets_lists, scores_lists):.5f}')
    plt.savefig(output_path.with_suffix('.eval.pcc.pdf')) 


    # ---------------------- section ---------------------- #
    ori_scores_lists = scores_lists.tolist()
    scores_lists = np.where(scores_lists > CUTOFF, 1, 0).tolist()

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        for chr, start, stop, tf, celltype, targets_list, ori_scores_list, scores_list in zip(chrs, starts, stops, tfs, celltypes, targets_lists.tolist(), ori_scores_lists, scores_lists):
            writer.writerow([chr, start, stop, tf, celltype, targets_list, ori_scores_list, scores_list])
    logger.info(
            f'auc: {metrics[0]:.5f}  '
            f'aupr: {metrics[1]:.5f}  '
            f'pcc: {metrics[2]:.5f}  '
            f'srcc: {metrics[3]:.5f}  '
            f'recall score: {metrics[4]:.5f}  '
            f'f1 score: {metrics[5]:.5f}  '
            f'accuracy: {metrics[6]:.5f}  '
            f'balanced accuracy: {metrics[7]:.5f}'
            )
    logger.info(f'Eval Completed')


def output_predict(chrs, starts, stops, scores_lists, tfs, celltypes, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path = output_path.with_suffix('.predict.tsv')

    ori_scores_lists = scores_lists.tolist()
    scores_lists = np.where(scores_lists > CUTOFF, 1, 0).tolist()

    with open(predict_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        #writer.writerow(['chr', 'start', 'stop', 'predict'])
        for chr, start, stop, tf, celltype, ori_scores_list, scores_list in zip(chrs, starts, stops, tfs, celltypes, ori_scores_lists, scores_lists):
            writer.writerow([chr, start, stop, tf, celltype, ori_scores_list, scores_list])
    logger.info(f'Predicting Completed')