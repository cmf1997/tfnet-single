import numpy as np
import pdb
from logzero import logger
from tfnet.all_tfs import all_tfs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
from tfnet.evaluation import get_auc, get_f1, get_accuracy_score, get_balanced_accuracy_score, get_recall, get_aupr, get_precision, get_pcc, get_srcc
import warnings
import sys


sys.setrecursionlimit(65520)
sns.set_theme(style="ticks")
warnings.filterwarnings("ignore",category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')
palette = ['#DC143C', '#4169E1','#ff69b4']

def read_predict(predict_file):
    with open(predict_file, 'r') as fp:
        chr_list = []
        start_list = []
        stop_list = []
        tf_list = []
        target_list = []
        ori_predict_list = []
        predict_list = []
        for line in fp:
            chr, start, stop, tf, target, ori_predict, predict = line.split('\t')
            tf_list.append(tf)
            target_list.append(int(target))
            ori_predict_list.append(float(ori_predict))
            predict_list.append(int(predict))
    return tf_list, target_list, ori_predict_list, predict_list

tf_list, target_list, ori_predict_list ,predict_list = read_predict("results/unseen/TFNet.unseen.eval.tsv")


eval_data = pd.DataFrame({
                "AUC" : get_auc(target_list, ori_predict_list),
                "AUPR": get_aupr(target_list, ori_predict_list),
                "ACCURACY": get_accuracy_score(target_list, np.array(ori_predict_list)),
                "BALANCED_ACCURACY": get_balanced_accuracy_score(target_list, np.array(ori_predict_list))
                }, index=[0])
plt.clf()
ax = sns.barplot(data=eval_data, alpha=0.7)
ax.set_ylim([0,1])
ax.tick_params(axis='x', labelrotation=45)
#plt.tick_params(axis='x', labelrotation=45)
ax.figure.savefig("results/TFNet.unseen.eval.pdf")


eval_data = pd.DataFrame({"TF" : tf_list,
                        "Target":target_list,
                        "Predict" : ori_predict_list
                        })

eval_data = eval_data.sort_values("TF")

all_eval_data = eval_data[['Target','Predict']]
all_eval_data['TF'] = 'All'
all_eval_data = all_eval_data[['TF', 'Target','Predict']]
all_eval_data = pd.concat([eval_data,all_eval_data], ignore_index=True)
plt.clf()
sns.lmplot(
    data=all_eval_data, x="Target", y="Predict",
    hue="TF", col="TF", height=4, palette=palette, scatter_kws={"alpha":0.5}
)
plt.savefig("results/TFNet.unseen.eval.pcc.pdf")

plt.clf()
fig, axs = plt.subplots(figsize=(10,5), ncols=2)
RocCurveDisplay.from_predictions(eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Target'], eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Predict'],ax=axs[0], name="TBP", lw=2, color = palette[0])
RocCurveDisplay.from_predictions(eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Target'], eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Predict'],ax=axs[0], name="MAFF",lw=2, color = palette[1])
PrecisionRecallDisplay.from_predictions(eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Target'], eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Predict'], ax=axs[1], name="TBP",lw=2, color = palette[0])
PrecisionRecallDisplay.from_predictions(eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Target'], eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Predict'], ax=axs[1], name="MAFF",lw=2, color = palette[1])

fig.savefig("results/eval.MAFF.TBP.auc.aupr.pdf")


get_pcc(eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Target'], eval_data[eval_data['TF']=='AAIFSGKLMGSQLYKPIVFV']['Predict'])
get_pcc(eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Target'], eval_data[eval_data['TF']=='VCKKSERAAGCRLKQRRRGY']['Predict'])
get_pcc(eval_data['Target'], eval_data['Predict'])


# ---------------------- classweight ---------------------- #
# ---------------------- classweight ---------------------- #
'''
pre_cutoffs = [0.4, 0.45, 0.5, 0.55, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

targets_array_weight, ori_predict_array_weight ,predict_array_weight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/weight/SimpleCNN_2d.eval.tsv")
targets_array_noweight, ori_predict_array_noweight ,predict_array_noweight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/noweight/SimpleCNN_2d.eval.tsv")

auc_weight = get_auc(targets_array_weight, ori_predict_array_weight)
auc_noweight = get_auc(targets_array_noweight, ori_predict_array_noweight)

aupr_weight = get_aupr(targets_array_weight, ori_predict_array_weight)
aupr_noweight = get_aupr(targets_array_noweight, ori_predict_array_noweight)

cutoffs = []
recall_weight = []
recall_noweight = []
precision_weight = []
precision_noweight = []
f1_weight = []
f1_noweight = []
balanced_accuracy_weight = []
balanced_accuracy_noweight = []

for cutoff in pre_cutoffs:
    cutoffs.append(cutoff)
    recall_weight.append(get_mean_recall(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    recall_noweight.append(get_mean_recall(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    precision_weight.append(get_mean_precision(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    precision_noweight.append(get_mean_precision(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    f1_weight.append(get_mean_f1(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    f1_noweight.append(get_mean_f1(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    balanced_accuracy_weight.append(get_mean_balanced_accuracy_score(targets_array_weight, ori_predict_array_weight, axis = 0, cutoff=cutoff))
    balanced_accuracy_noweight.append(get_mean_balanced_accuracy_score(targets_array_noweight, ori_predict_array_noweight, axis = 0, cutoff=cutoff))

cutoff_data = pd.DataFrame({"CUTOFF" : cutoffs,
                        "RECALL_classweight":recall_weight,
                        "RECALL" : recall_noweight,
                        "PRECISION_classweight" : precision_weight,
                        "PRECISION" : precision_noweight,
                        "F1_classweight": f1_weight,
                        "F1" : f1_noweight,
                        "BALANCE_ACCURACY_classweight" : balanced_accuracy_weight,
                        "BALANCE_ACCURACY" : balanced_accuracy_noweight
                        })

cutoff_data = cutoff_data.melt(id_vars=['CUTOFF'],var_name="Model", value_name="value")
cutoff_data['type'] = [ i.split("_")[0] for i in cutoff_data['Model'].tolist() ]

sns.relplot(
    data=cutoff_data,
    x="CUTOFF", y="value",
    hue="Model",
    kind='line', 
    col="type", col_wrap=2,
    palette=palette[:2],
    height=4,
    aspect=1,
    lw=6, alpha = 0.7,
    facet_kws={'sharey': False, 'sharex': False}
)
plt.title("for each prediction    for each TFs")
plt.xlabel("CUTOFF")
plt.ylabel("Value")
plt.savefig("results/eval.GM12878.compare.classweight.pdf")
'''


# ---------------------- pseudosequences tfnet ---------------------- #
# ---------------------- pseudosequences tfnet ---------------------- #


# ---------------------- load data ---------------------- #
targets_array_tfnet, ori_predict_array_tfnet ,predict_array_tfnet = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/tfnet/TFNet.eval.tsv")
targets_array_weight, ori_predict_array_weight ,predict_array_weight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/weight/SimpleCNN_2d.eval.tsv")
targets_array_noweight, ori_predict_array_noweight ,predict_array_noweight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/noweight/SimpleCNN_2d.eval.tsv")

# ---------------------- auc aupr ---------------------- #
#RocCurveDisplay.from_predictions(targets_array_tfnet[:,1], ori_predict_array_tfnet[:,1])

auc_data = pd.DataFrame({"TF" : all_tfs,
                        "AUC_TFNet" : get_auc(targets_array_tfnet, ori_predict_array_tfnet),
                        "AUC_Weight":get_auc(targets_array_weight, ori_predict_array_weight),
                        "AUC" : get_auc(targets_array_noweight, ori_predict_array_noweight)
                        }).melt(id_vars=['TF'],var_name="Model", value_name="value")

sns.barplot(auc_data, x="TF", y="value", hue="Model", dodge=True, alpha=0.8, palette=palette[:3])
plt.xlabel("")
plt.ylabel("AUC")
plt.tick_params(axis='x', labelrotation=45)
plt.savefig("results/eval.GM12878.auc.pdf")
plt.show()


aupr_data = pd.DataFrame({"TF" : all_tfs,
                        "AUPR_TFNet" : get_aupr(targets_array_tfnet, ori_predict_array_tfnet),
                        "AUPR_Weight":get_aupr(targets_array_weight, ori_predict_array_weight),
                        "AUPR" : get_aupr(targets_array_noweight, ori_predict_array_noweight)
                        }).melt(id_vars=['TF'],var_name="Model", value_name="value")
plt.clf()
sns.barplot(aupr_data, x="TF", y="value", hue="Model", dodge=True, alpha=0.8, palette=palette[:3])
plt.xlabel("")
plt.ylabel("AUPR")
plt.tick_params(axis='x', labelrotation=45)
plt.savefig("results/eval.GM12878.aupr.pdf")


# ---------------------- eval ---------------------- #
pre_cutoffs = [0.4, 0.45, 0.5, 0.55, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

cutoffs = []

recall_tfnet = []
recall_weight = []
recall_noweight = []

precision_tfnet = []
precision_weight = []
precision_noweight = []

f1_tfnet = []
f1_weight = []
f1_noweight = []

balanced_accuracy_tfnet = []
balanced_accuracy_weight = []
balanced_accuracy_noweight = []

for cutoff in pre_cutoffs:
    cutoffs.append(cutoff)

    recall_tfnet.append(get_mean_recall(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    recall_weight.append(get_mean_recall(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    recall_noweight.append(get_mean_recall(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    precision_tfnet.append(get_mean_precision(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    precision_weight.append(get_mean_precision(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    precision_noweight.append(get_mean_precision(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    f1_tfnet.append(get_mean_f1(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    f1_weight.append(get_mean_f1(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    f1_noweight.append(get_mean_f1(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    balanced_accuracy_tfnet.append(get_mean_balanced_accuracy_score(targets_array_tfnet, ori_predict_array_tfnet, axis = 0, cutoff=cutoff))
    balanced_accuracy_weight.append(get_mean_balanced_accuracy_score(targets_array_weight, ori_predict_array_weight, axis = 0, cutoff=cutoff))
    balanced_accuracy_noweight.append(get_mean_balanced_accuracy_score(targets_array_noweight, ori_predict_array_noweight, axis = 0, cutoff=cutoff))

cutoff_data = pd.DataFrame({"CUTOFF" : cutoffs,
                        "RECALL_tfnet" : recall_tfnet,
                        "RECALL_classweight":recall_weight,
                        "RECALL" : recall_noweight,

                        "PRECISION_tfnet" : precision_tfnet,
                        "PRECISION_classweight" : precision_weight,
                        "PRECISION" : precision_noweight,

                        "F1_tfnet" : f1_tfnet,
                        "F1_classweight": f1_weight,
                        "F1" : f1_noweight,

                        "BALANCE_ACCURACY_tfnet" : balanced_accuracy_tfnet,
                        "BALANCE_ACCURACY_classweight" : balanced_accuracy_weight,
                        "BALANCE_ACCURACY" : balanced_accuracy_noweight
                        })

cutoff_data = cutoff_data.melt(id_vars=['CUTOFF'],var_name="Model", value_name="value")
cutoff_data['type'] = [ i.split("_")[0] for i in cutoff_data['Model'].tolist() ]

# ---------------------- plot ---------------------- #
sns.relplot(
    data=cutoff_data,
    x="CUTOFF", y="value",
    hue="Model",
    kind='line', 
    col="type", col_wrap=2,
    palette=palette[:3],
    height=4,
    aspect=1,
    lw=6, alpha = 0.7,
    facet_kws={'sharey': False, 'sharex': False}
)
plt.title("pseudosequence TFNet")
plt.savefig("results/eval.GM12878.compare.tfnet.pdf")
