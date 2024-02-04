#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : datasets.py
@Time : 2023/11/09 11:19:37
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tfnet.data_utils import ACIDS
from tfnet.all_tfs import all_tfs
import re
import pysam
from pathlib import Path
import pyBigWig
import pdb


__all__ = ["TFBindDataset"]


# code
class TFBindDataset(Dataset):
    def __init__(self, data_list, genome_fasta_file, mappability_bw_file, chromatin_bw_files, DNA_len=1024, DNA_pad=10, tf_len=20, padding_idx=0, target_len=200, DNA_N = True):
        self.DNA_N = DNA_N
        self.data_list = data_list
        self.DNA_x, self.tf_x, self.targets = [], [], []
        self.genome_fasta = pysam.Fastafile(genome_fasta_file)

        #self.bigwig_data = {}
        #for index, single_bw_file in enumerate(bw_file):
        #    self.bigwig_data[index] = pyBigWig.open(single_bw_file)

        self.mappability_bw =  pyBigWig.open(mappability_bw_file)

        self.chromatin_bw = {}
        for single_bw_file in chromatin_bw_files:
            celltype = str(Path(single_bw_file).stem)
            self.chromatin_bw[celltype] = single_bw_file


        self.DNA_pad = DNA_pad
        self.DNA_len = DNA_len
        self.tf_len = tf_len
        self.bind_list = [ i[-3] for i in data_list]
        self.bind_list = np.asarray(self.bind_list, dtype=np.float32)

    def __getitem__(self, idx):
        chr, start, stop, bind_list, tf, celltype= self.data_list[idx]

        bind_list = np.asarray(bind_list, dtype=np.float32)
        start = int(start)
        stop = int(stop)

        celltype = celltype.rstrip('\n')
        # ---------------------- shift ---------------------- #
        shift = np.random.randint(-20, 20+1)
        start += shift
        stop += shift

        DNA_seq = self.genome_fasta.fetch(chr, start, stop)
        if self.DNA_N:
            d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3, 'N':4, 'n':4}
            DNA_seq = self.DNA_pad*"N" + DNA_seq + self.DNA_pad*"N"     # for DNA pad to set conv1d output same dim
            mat = np.zeros((len(DNA_seq),5))
            for i in range(len(DNA_seq)):
                mat[i,d[DNA_seq[i]]] = 1
            DNA_x = mat[:self.DNA_len + self.DNA_pad*2, :5]
        else: 
            d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3}
            mat = np.zeros((len(DNA_seq),4))
            for i in range(len(DNA_seq)):
                if len(re.findall('[atcg]', DNA_seq[i].lower())) != 0:  # no one hot for n
                    mat[i,d[DNA_seq[i]]] = 1
            DNA_x = mat[:self.DNA_len, :4]
        DNA_rc_x = DNA_x[:,::-1].copy()
        DNA_x = torch.tensor(DNA_x, dtype=torch.float32)
        DNA_rc_x = torch.tensor(DNA_rc_x, dtype=torch.float32)
        DNA_x = torch.cat([DNA_x, DNA_rc_x], dim=1)


        # ---------------------- bw_list need padding like DNA_x ---------------------- #
        '''
        bigwig_signals = []
        bigwig_signals_rc = []
        
        for index in range(len(self.bigwig_data)):
            bigwig_signal = np.array(self.bigwig_data[index].values(chr,start,stop))
            bigwig_signal[np.isnan(bigwig_signal)] = 0
            bigwig_signals.append(bigwig_signal)
            bigwig_signals_rc.append(bigwig_signal[::-1].copy())
        '''


        # ---------------------- celltype ---------------------- #
        bigwig_signals = []
        bigwig_signals_rc = []    

        mappability_bw = self.mappability_bw.values(chr,start,stop)
        #mappability_bw[np.isnan(mappability_bw)] = 0
        np.array(mappability_bw)[np.isnan(mappability_bw)] = 0

        bigwig_signals.append(mappability_bw)
        bigwig_signals_rc.append(mappability_bw[::-1].copy())

        #pdb.set_trace()
        chromatin_bw =  pyBigWig.open(self.chromatin_bw[celltype]).values(chr,start,stop)
        #chromatin_bw[np.isnan(chromatin_bw)] = 0
        np.array(chromatin_bw)[np.isnan(chromatin_bw)] = 0

        bigwig_signals.append(chromatin_bw)
        bigwig_signals_rc.append(chromatin_bw[::-1].copy())


        # ---------------- concatenate rc, comment to abort----------------#
        bigwig_signals.extend(bigwig_signals_rc)
        for i in range(len(bigwig_signals)):
            if self.DNA_N:
                bigwig_signal = [0 for i in range(self.DNA_pad)] + [j for j in bigwig_signals[i]] + [0 for i in range(self.DNA_pad)]
            else:
                bigwig_signal = bigwig_signals[i]

            bigwig_signal = np.expand_dims(bigwig_signal, axis=-1)
            bigwig_signal = torch.tensor(bigwig_signal, dtype=torch.float32)

            DNA_x = torch.cat([DNA_x, bigwig_signal],dim=1)
            

        tf_x = [ACIDS.index(x if x in ACIDS else "-") for x in tf]
        assert len(tf_x) == self.tf_len
        
        tf_x = torch.tensor(tf_x, dtype=torch.long)

        return (DNA_x, tf_x), bind_list
    def __len__(self):
        return len(self.data_list)
