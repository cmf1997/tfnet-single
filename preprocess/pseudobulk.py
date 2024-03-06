#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : Untitled-1
@Time : 2024/02/28 10:40:19
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# code
T_umap = pd.read_csv("/Users/cmf/Downloads/GSE129785_scATAC-TME-TCells.cell_barcodes.txt.gz", sep='\t', compression='gzip', header=0)

sns.scatterplot(data=T_umap, x='UMAP1', y='UMAP2',hue='Clusters')
plt.show()


# compared with https://bio.liclab.net/scATAC-Ref/browseDetail?sample=GSE129785_TME_All

cluster1: Naive CD4
cluster2: Th17
cluster3: Tfh
cluster4: Treg
cluster5: Naive CD8 T
cluster6: 
cluster7: Memory CD8 T
cluster8: Exhausted CD8 T
cluster9: Effector CD8 T



# raw data should include 
set(T_umap[T_umap['Clusters'].isin(['Cluster5','Cluster7','Cluster8','Cluster9'])]['Group'])
{'SU001_Total_Pre', 'SU006_Immune_Pre', 'SU006_Tcell_Pre', 'SU005_Total_Post', 
 'SU008_Immune_Pre', 'SU008_Tcell_Post', 'SU001_Total_Post2', 'SU006_Total_Post', 
 'SU008_Immune_Post', 'SU008_Tcell_Pre', 'SU009_Tumor_Immune_Pre', 'SU007_Total_Post', 
 'SU010_Total_Post', 'SU001_Tcell_Post', 'SU001_Immune_Post2', 'SU009_Tcell_Pre', 
 'SU009_Tumor_Immune_Post', 'SU001_Tcell_Post2', 'SU009_Tcell_Post', 
 'SU001_Tumor_Immune_Post', 'SU010_Total_Pre'}

# for example SU008_Tcell_Pre https://www.ncbi.nlm.nih.gov/sra?term=SRX5679932
# ~/software/sratoolkit.3.0.0-centos_linux64/bin/prefetch SRR8893788

# after cellranger, bamfile read length including R1 150-16-10 = 124, R2 150 using for align genome
# using /Users/cmf/Desktop/Analysis/scATAC/Pseudobulk.sh to extract related read into cluster specific bam
# generate bw from bam file as chromatin accessibility


