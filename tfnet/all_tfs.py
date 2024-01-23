#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : all_tfs.py
@Time : 2023/11/09 11:19:01
@Author : Cmf
@Version : 1.0
@Desc : 
for tfs names
ls *.bed.gz | sed 's/Chip_K562_//g' | sed 's/.bed.gz//g' | sed 's/\t/"/g' | sed 's/^/"&/g' | sed 's/$/"&/g' | sed ':a;N;$!ba;s/\n/,/g'
the tfs order should be the same as data/tf_chip/chip.txt
'''


# code
__all__ = ['all_tfs', 'all_tfs_GM12878', 'all_tfs_K562', 'all_tfs_H1ESC', 'shared_tfs', 'shared_tfs_dbd']

all_tfs_original = [
    "ARID3A","ATF3","ATF7","CEBPB","CREB1",
    "CTCF","E2F1","E2F6","EGR1","FOXA1",
    "FOXA2","GABPA","GATA3","HNF4A","JUND","MAFK",
    "MAX","MYC","NANOG","REST","RFX5","SPI1",
    "SRF","STAT3","TCF12","TCF7L2",
    "TEAD4","YY1","ZNF143"
]


all_tfs_GM12878 = [
    "ATF3", 
    "BHLHE40", 
    "CEBPB", 
    "CTCF", 
    "E2F4", 
    "EGR1", # check
    "ELF1", 
    "ELK1", 
    "ETS1", 
    "FOS", 
    "JUND", 
    "MAX",  # check
    "MEF2A", 
    "MYC", 
    "NFE2", # abort
    "NFYA", 
    "NFYB", 
    "NRF1", 
    "REST", 
    "SPI1", # check
    "RAD21", 
    "RFX5", # check
    "SIX5", 
    "SMC3", 
    "SP1", 
    "SRF", 
    "STAT1",
    "STAT5A", 
    "NR2C2", 
    "USF2", 
    "YY1", # check
    "ZBTB33", 
    "ZNF143" # check
]
# [ "ATF7","CREB1","E2F1","EGR1","EP300","GABPA","MAFK","MAX","RFX5","SPI1","SRF","TAF1","TCF12","YY1","ZNF143" ] for factornet 

all_tfs_K562 = [
    "ARID3A","ATF1","ATF3","BACH1","BHLHE40","CCNT2","CEBPB","CTCF",
    "CTCFL","E2F4","E2F6","EFOS","EGATA","EGR1","EJUNB","EJUND","ELF1",
    "ELK1","ETS1","FOS","FOSL1","GABP","GATA1","GATA2","IRF1","JUN",
    "JUND","MAFF","MAFK","MAX","MEF2A","MYC","NFE2","NFYA","NFYB","NR2F2",
    "NRF1","PU1","RAD21","REST","RFX5","SIX5","SMC3","SP1","SP2","SRF",
    "STAT1","STAT2","STAT5A","TAL1","TBP","THAP1","TR4","USF1","USF2","YY1",
    "ZBTB33","ZBTB7A","ZNF143","ZNF263"
]


all_tfs_H1ESC = [
#all_tfs = [
    "ATF3","BACH1","BRCA1","CEBPB","CTCF","EGR1",
    "FOSL1","GABP","JUN","JUND","MAFK","MAX","MYC",
    "NRF1","P300","POU5F1","RAD21","REST","RFX5",
    "RXRA","SIX5","SP1","SP2","SP4","SRF","TBP",
    "TCF12","USF1","USF2","YY1","ZNF143"
]


shared_tfs = [
    'ATF3', 'CEBPB', 'CTCF', 'EGR1', 'JUND', 'MAX', 
    'MYC', 'NRF1', 'RAD21', 'REST', 'RFX5', 'SIX5', 
    'SP1', 'SRF', 'USF2', 'YY1', 'ZNF143']


# ---------------------- no dbd info ---------------------- #
no_dbd = ['CCNT2', 'P300', 'RAD21', 'SMC3']


#shared_tfs_dbd = [
all_tfs = [
    'CEBPB', 'CTCF', 'EGR1', 'JUND', 'MAX', 
    'MYC', 'NRF1', 'REST', 'RFX5', 'SIX5', 
    'SP1', 'SRF', 'USF2', 'YY1', 'ZNF143']