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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from tfnet.all_tfs import all_tfs
from logzero import logger
import pdb
import warnings
warnings.filterwarnings('error')  

__all__ = ['CUTOFF', 'get_auc', 'get_aupr', 'get_recall', 'get_f1', 'get_precision', 'get_balanced_accuracy_score','get_label_ranking_average_precision_score', 'get_group_metrics', 'output_eval', 'output_predict']

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


def output_eval(chrs, starts, stops, targets_lists, scores_lists, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path = output_path.with_suffix('.eval.tsv')

    metrics = []
    metrics.append(get_auc(targets_lists, scores_lists))
    metrics.append(get_aupr(targets_lists, scores_lists))
    metrics.append(get_recall(targets_lists, scores_lists))
    metrics.append(get_f1(targets_lists, scores_lists))
    metrics.append(get_accuracy_score(targets_lists, scores_lists))
    metrics.append(get_balanced_accuracy_score(targets_lists, scores_lists))


    # ---------------------- plot ---------------------- #
    plot_data = pd.DataFrame({
        "TF_name" : all_tfs,
        "AUC" : get_auc(targets_lists, scores_lists),
        "AUPR" : get_aupr(targets_lists, scores_lists),
        "RECALL" : get_recall(targets_lists, scores_lists),
        "F1" : get_f1(targets_lists, scores_lists)
        }
    )

    plot_data.to_csv(output_path.with_suffix('.eval.repl.tsv'), sep='\t')

    #sns.set(rc={'figure.figsize':(7,4)})
    rel_plot = sns.scatterplot(data=plot_data, x="AUC", y="AUPR", hue="RECALL", size="F1", sizes=(50,200))
    sns.move_legend(rel_plot, "upper left", bbox_to_anchor=(1, 0.75))
    fig = rel_plot.get_figure()
    fig.savefig(output_path.with_suffix('.eval.repl.pdf')) 


    fig, axes = plt.subplots(2, 2)
    xlabel = all_tfs
    sns.barplot(data=plot_data, x='TF_name', y='AUC', ax=axes[0,0])
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].set_xticklabels(xlabel, fontsize=4)
    axes[0,0].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='AUPR', ax=axes[0,1])
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].set_xticklabels(xlabel, fontsize=4)
    axes[0,1].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='RECALL', ax=axes[1,0])
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].set_xticklabels(xlabel, fontsize=4)
    axes[1,0].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='F1', ax=axes[1,1])
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].set_xticklabels(xlabel, fontsize=4)
    axes[1,1].set(xlabel='')

    fig.savefig(output_path.with_suffix('.eval.box.pdf')) 
    # ---------------------- section ---------------------- #
    ori_scores_lists = scores_lists
    ori_scores_lists = np.split(ori_scores_lists,ori_scores_lists.shape[0], axis=0)
    ori_scores_lists = [i.flatten().tolist() for i in ori_scores_lists]

    scores_lists = np.where(scores_lists > CUTOFF, 1, 0)
    scores_lists = np.split(scores_lists,scores_lists.shape[0], axis=0)
    scores_lists = [i.flatten().tolist() for i in scores_lists]

    targets_lists = [ list(map(int,i.tolist())) for i in targets_lists ]

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        #writer.writerow(['chr', 'start', 'stop', 'targets', 'predict'])
        for chr, start, stop, targets_list, ori_scores_list, scores_list in zip(chrs, starts, stops, targets_lists, ori_scores_lists, scores_lists):
            writer.writerow([chr, start, stop, targets_list, ori_scores_list, scores_list])
    logger.info(
            f'auc: {metrics[0]:.5f}  '
            f'aupr: {metrics[1]:.5f}  '
            f'recall score: {metrics[2]:.5f}  '
            f'f1 score: {metrics[3]:.5f}  '
            f'accuracy: {metrics[4]:.5f}  '
            f'balanced accuracy: {metrics[5]:.5f}'
            )
    logger.info(f'Eval Completed')


def output_predict(chrs, starts, stops, scores_lists, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path = output_path.with_suffix('.predict.tsv')

    scores_lists = np.where(scores_lists > CUTOFF, 1, 0)
    scores_lists = np.split(scores_lists,scores_lists.shape[0], axis=0)
    scores_lists = [i.flatten().tolist() for i in scores_lists]

    with open(predict_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(['chr', 'start', 'stop', 'predict'])
        for chr, start, stop, scores_list in zip(chrs, starts, stops, scores_lists):
            writer.writerow([chr, start, stop, scores_list])
    logger.info(f'Predicting Completed')