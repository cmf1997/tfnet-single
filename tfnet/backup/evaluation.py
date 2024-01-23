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
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from tfnet.all_tfs import all_tfs
from logzero import logger
import pdb

__all__ = ['CUTOFF', 'get_mean_auc', 'get_mean_recall', 'get_mean_aupr', 'get_mean_f1', 'get_mean_accuracy_score', 'get_mean_balanced_accuracy_score','get_label_ranking_average_precision_score', 'get_group_metrics', 'output_eval', 'output_predict']

CUTOFF = 0.8


# code
def get_mean_auc(targets, scores):
    auc_scores = get_auc(targets, scores)
    return np.mean(auc_scores)

def get_auc(targets, scores):
    auc_scores = []
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], scores[:, i] )
        auc_scores.append(auc)
    return auc_scores


def get_recall(targets, scores, cutoffs):
    recall_list = []
    for i in range(targets.shape[1]):
        recall = recall_score(targets[:, i], scores[:, i]> cutoffs[i], zero_division=1.0)
        recall_list.append(recall)
    return recall_list


def get_mean_recall(targets, scores, cutoffs):
    recall_list = get_recall(targets, scores, cutoffs)
    return np.mean(recall_list)


def get_aupr(targets, scores):
    aupr_list = []
    for i in range(targets.shape[1]):
        precisions, recalls, thresholds = precision_recall_curve(targets[:, i], scores[:, i])
        auc_precision_recall = auc(recalls, precisions)
        aupr_list.append(auc_precision_recall)
    return aupr_list


def get_mean_aupr(targets, scores):
    aupr_list = get_aupr(targets, scores)
    return np.mean(aupr_list)


def get_label_ranking_average_precision_score(targets, scores):
    return label_ranking_average_precision_score(targets, scores)


def get_f1(targets, scores):
    f1_list = []
    cutoffs = []
    pre_cutoffs = [0.55,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    for i in range(targets.shape[1]):
        pre_f1_list = []
        for j in pre_cutoffs:
            pre_f1_list.append(f1_score(targets[:, i], scores[:, i] > j, zero_division=1.0))
        cutoffs.append(pre_cutoffs[np.argmax(pre_f1_list)])
        f1_list.append(np.max(pre_f1_list))
    return f1_list, cutoffs


def get_mean_f1(targets, scores):
    f1_list, cutoffs = get_f1(targets, scores)
    return np.mean(f1_list)


def get_mean_accuracy_score(targets, scores, cutoffs):
    accuracy_score_list = []
    for i in range(targets.shape[1]):
        accuracy = accuracy_score(targets[:, i], scores[:, i] > cutoffs[i])
        accuracy_score_list.append(accuracy)
    return np.mean(np.array(accuracy_score_list, dtype=float))


def get_mean_balanced_accuracy_score(targets, scores, cutoffs):
    accuracy_score_list = []
    for i in range(targets.shape[1]):
        accuracy = balanced_accuracy_score(targets[:, i], scores[:, i] > cutoffs[i])
        accuracy_score_list.append(accuracy)
    return np.mean(np.array(accuracy_score_list, dtype=float))


def output_eval(chrs, starts, stops, targets_lists, scores_lists, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path = output_path.with_suffix('.eval.tsv')

    metrics = []

    f1_list, cutoffs = get_f1(targets_lists, scores_lists)
    # ---------------------- save cutoffs for prediction ---------------------- #
    logger.info(f'Save cutoffs for Prediction')
    with open(output_path.with_suffix('.eval.cutoffs'), "w") as f:
        for s in cutoffs:
            f.write(str(s) +"\n")

    metrics.append(get_mean_auc(targets_lists, scores_lists))
    metrics.append(get_mean_aupr(targets_lists, scores_lists))

    metrics.append(get_mean_recall(targets_lists, scores_lists, cutoffs)) #
    metrics.append(np.mean(f1_list)) 

    metrics.append(get_label_ranking_average_precision_score(targets_lists, scores_lists))

    metrics.append(get_mean_accuracy_score(targets_lists, scores_lists, cutoffs)) #
    metrics.append(get_mean_balanced_accuracy_score(targets_lists, scores_lists, cutoffs)) #

    # ---------------------- plot ---------------------- #
    plot_data = pd.DataFrame({
        "TF_name" : all_tfs,
        "AUC" : get_auc(targets_lists, scores_lists),
        "AUPR" : get_aupr(targets_lists, scores_lists),
        "RECALL" : get_recall(targets_lists, scores_lists, cutoffs),  #
        "F1" : f1_list #
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

    scores_lists = np.split(scores_lists,scores_lists.shape[1], axis=1)
    scores_lists_binary = []
    for index, i in enumerate(scores_lists):
        tmp_list = np.where(i > cutoffs[index], 1, 0)
        scores_lists_binary.append(tmp_list.flatten().tolist())
    scores_lists_array = np.array(scores_lists_binary).transpose(1,0)
    #scores_lists_array = np.squeeze(scores_lists_array,-1).transpose(1,0)

    scores_lists_array = np.split(scores_lists_array,scores_lists_array.shape[0], axis=0)
    scores_lists_array = [i.flatten().tolist() for i in scores_lists_array]


    targets_lists = [ list(map(int,i.tolist())) for i in targets_lists ]

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        #writer.writerow(['chr', 'start', 'stop', 'targets', 'ori_predict', 'predict'])
        for chr, start, stop, targets_list, ori_scores_list, scores_list in zip(chrs, starts, stops, targets_lists, ori_scores_lists, scores_lists_array):
            writer.writerow([chr, start, stop, targets_list, ori_scores_list, scores_list])
    
    logger.info(
            f'mean_auc: {metrics[0]:.5f}  '
            f'aupr: {metrics[1]:.5f}  '
            f'recall score: {metrics[2]:.5f}  '
            f'f1 score: {metrics[3]:.5f}  '
            f'lrap: {metrics[4]:.5f}  '
            f'accuracy: {metrics[5]:.5f}  '
            f'balanced accuracy: {metrics[6]:.5f}'
            )
    logger.info(f'Eval Completed')


def output_predict(chrs, starts, stops, scores_lists, output_path: Path):

    # ---------------------- load cutoffs from output_eval ---------------------- #
    if output_path.with_suffix('.eval.cutoffs').exists():
        cutoffs = []
        with open(output_path.with_suffix('.eval.cutoffs'), "r") as f:
            for line in f:
                cutoffs.append(float(line.strip()))
    else:
        logger.info(f'Run Eval to determine dynamic cutoffs')


    output_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path = output_path.with_suffix('.predict.tsv')


    scores_lists = np.split(scores_lists,scores_lists.shape[1], axis=1)
    scores_lists_binary = []
    for index, i in enumerate(scores_lists):
        tmp_list = np.where(i > cutoffs[index], 1, 0)
        scores_lists_binary.append(tmp_list.flatten().tolist())
    scores_lists_array = np.array(scores_lists_binary).transpose(1,0)
    #scores_lists_array = np.squeeze(scores_lists_array,-1).transpose(1,0)

    scores_lists_array = np.split(scores_lists_array,scores_lists_array.shape[0], axis=0)
    scores_lists_array = [i.flatten().tolist() for i in scores_lists_array]


    with open(predict_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(['chr', 'start', 'stop', 'predict'])
        for chr, start, stop, scores_list in zip(chrs, starts, stops, scores_lists_array):
            writer.writerow([chr, start, stop, scores_list])
    logger.info(f'Predicting Completed')