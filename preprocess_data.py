#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : preprocess_chip_data.py
@Time : 2023/11/13 15:20:52
@Author : Cmf
@Version : 1.0
@Desc : preprocess atac and chip-seq data, output as a DNA sequence,  Chromatin accessibility and multiple binary labels
blacklist were downloaded from https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg19-blacklist.v2.bed.gz
test data were downloaded from https://www.synapse.org/#!Synapse:syn6131484/wiki/402033
EP300,TAF1 were removed due to absent of DBD

concat all train data and leave MAFF TBP for eval
dbd info no CCNT2, P300, RAD21, SMC3
and 
p300 - ep300
pu1 - spi1
NR2C2 - TR4
are the same 
rm CCNT2, P300, RAD21, SMC3, pu1, NR2C2, p300

ls *.data_train.txt.gz | grep -v MAFF.data_train.txt.gz | grep -v TBP.data_train.txt.gz | xargs -p gunzip -c > all.data_train.txt
ls *.data_valid.txt.gz | grep -E 'MAFF|TBP' | grep valid.txt.gz | xargs -p gunzip -c > select.data_valid.txt

or
ls *.txt.gz | grep -v MAFF | grep -v TBP | xargs -p gunzip -c > all.data.txt


'''

# here put the import lib
from ruamel.yaml import YAML
from pybedtools import BedTool, Interval
import pybedtools
import numpy as np
import parmap
import click
from pathlib import Path
import csv
from tfnet.all_tfs import all_tfs
import os

import pdb

# code
def get_genome_bed(genome_sizes_file):
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    genome_bed = genome_bed.sort()
    return chroms, chroms_sizes, genome_bed


def make_blacklist(blacklist_file, genome_sizes_file, genome_window_size):
    blacklist = BedTool(blacklist_file)
    blacklist = blacklist.slop(g=genome_sizes_file, b=genome_window_size)
    # Add ends of the chromosomes to the blacklist
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    blacklist2 = []
    for chrom, size in zip(chroms, chroms_sizes):
        blacklist2.append(Interval(chrom, 0, genome_window_size))
        blacklist2.append(Interval(chrom, size - genome_window_size, size))
    blacklist2 = BedTool(blacklist2)
    blacklist = blacklist.cat(blacklist2)
    return blacklist


def get_chip_beds(input_dir):
    print(f'load chip from {input_dir}/chip.txt')
    chip_info_file = input_dir + '/chip.txt'
    chip_info = np.loadtxt(chip_info_file, dtype=str)
    if len(chip_info.shape) == 1:
        chip_info = np.reshape(chip_info, (-1,len(chip_info)))
    tfs = list(chip_info[:, 1])
    chip_bed_files = [input_dir + '/' + i for i in chip_info[:,0]]
    chip_beds = [BedTool(chip_bed_file) for chip_bed_file in chip_bed_files]
    # ---------------------- Sorting BED files ---------------------- # 
    chip_beds = [chip_bed.sort() for chip_bed in chip_beds]
    return tfs, chip_beds


def load_chip_multiTask(input_dir, genome_sizes_file, target_window_size, genome_window_size, genome_window_step, blacklist):
    tfs, chip_beds = get_chip_beds(input_dir)
    print(f'processing chip for positive and negative window')
    # ---------------------- Removing peaks outside of X chromosome and autosomes ---------------------- #
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    
    # ---------------------- genome windows ---------------------- #
    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=target_window_size,
                                            s=genome_window_step)
    genome_windows = genome_windows.sort()
    
    # ---------------------- for each tf ---------------------- #
    positive_windows_list = []
    #negative_windows_list = []
    nonnegative_regions_list = []

    for chip_bed in chip_beds:

        chip_bed = chip_bed.intersect(genome_bed, u=True, sorted=True)

        positive_windows = genome_windows.intersect(chip_bed, u=True, f=1.0*(target_window_size/2+1)/target_window_size, sorted=True)
        positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)
        positive_windows_list.append(positive_windows)

        chip_slop_bed = chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
        nonnegative_regions_bed = chip_slop_bed.cat(blacklist)
        nonnegative_regions_list.append(nonnegative_regions_bed)

        #negative_windows = genome_windows.intersect(nonnegative_regions_bed, wa=True, v=True, sorted=True)
        #negative_windows_list.append(negative_windows)

    #return tfs, positive_windows_list ,negative_windows_list, nonnegative_regions_list
    return tfs, positive_windows_list, nonnegative_regions_list


def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False


def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)


# ---------------------- make_features_multiTask ---------------------- #
def write_result(filename, tfs_bind_data, result_filefolder, prefix):
    with open(result_filefolder + prefix + '.' + filename +'.txt', 'w') as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for chr, start, stop, target_array, tf in tfs_bind_data:
            writer.writerow([chr, start, stop, target_array, tf])


def make_features_SingleTask(genome_sizes_file, tf, positive_windows, target_window_size, genome_window_size, nonnegative_regions, valid_chroms, test_chroms, result_filefolder):
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    genome_bed_train, genome_bed_valid, genome_bed_test = [subset_chroms(chroms_set, genome_bed) for chroms_set in (train_chroms, valid_chroms, test_chroms)]

    positive_windows_train = []
    positive_windows_valid = []
    positive_windows_test = []
    positive_data_train = []
    positive_data_valid = []
    positive_data_test = []

    positive_target = 1
    
    print(f'Splitting positive windows into training, validation, and testing sets for {tf}')
    
    for positive_window in positive_windows:
        if len(positive_window.chrom) > 8:
            pdb.set_trace()
        chrom = positive_window.chrom
        start = int(positive_window.start)
        stop = int(positive_window.stop)

        med = int((start + stop) / 2)
        start = int(med - int(genome_window_size) / 2)
        stop = int(med + int(genome_window_size) / 2)

        if chrom in test_chroms:
            positive_windows_test.append(positive_window)
            positive_data_test.append((chrom, start, stop, positive_target, tf))
        elif chrom in valid_chroms:
            positive_windows_valid.append(positive_window)
            positive_data_valid.append((chrom, start, stop, positive_target, tf))
        else:
            positive_windows_train.append(positive_window)
            positive_data_train.append((chrom, start, stop, positive_target, tf))

        
    b_size = (genome_window_size - target_window_size)/2
    positive_windows = positive_windows.slop(g=genome_sizes_file, b=b_size)

    positive_windows_train = BedTool(positive_windows_train)
    positive_windows_train = positive_windows_train.slop(g=genome_sizes_file, b=b_size)

    positive_windows_valid = BedTool(positive_windows_valid)
    positive_windows_valid = positive_windows_valid.slop(g=genome_sizes_file, b=b_size)

    positive_windows_test = BedTool(positive_windows_test)
    positive_windows_test = positive_windows_test.slop(g=genome_sizes_file, b=b_size)
    

    print('Getting negative training examples')
    negative_windows_train = positive_windows_train.shuffle(g=genome_sizes_file, # for single epoch
                                                            incl=genome_bed_train.fn,
                                                            excl=nonnegative_regions.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647)
                                                            #, output= 'negative_train_windows.bed'
                                                            )
    print('Getting negative validation examples')
    negative_windows_valid = positive_windows_valid.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_valid.fn,
                                                            excl=nonnegative_regions.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
    print('Getting negative testing examples')
    negative_windows_test = positive_windows_test.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_test.fn,
                                                            excl=nonnegative_regions.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))

    # Train
    print('Extracting data from negative training BEDs')
    negative_target = 0

    negative_data_train = [(window.chrom, window.start, window.stop, negative_target, tf)
                           for window in negative_windows_train]

    # Validation
    print('Extracting data from negative validation BEDs')
    negative_data_valid = [(window.chrom, window.start, window.stop, negative_target, tf)
                           for window in negative_windows_valid]
    
    # Test
    print('Extracting data from negative testing BEDs')
    negative_data_test = [(window.chrom, window.start, window.stop, negative_target, tf)
                           for window in negative_windows_test]


    data_train = negative_data_train + positive_data_train
    data_valid = negative_data_valid + positive_data_valid
    data_test = negative_data_test + positive_data_test

    print('Shuffling training data')
    np.random.shuffle(data_train)

    # ---------------------- write result ---------------------- #
    write_result("data_train", data_train, result_filefolder, prefix = tf)
    write_result("data_valid", data_valid, result_filefolder, prefix = tf)
    write_result("data_test", data_test, result_filefolder, prefix = tf)


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
def main(data_cnf, model_cnf):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    input_dir = data_cnf['input_dir']
    
    genome_window_size = model_cnf['padding']['DNA_len']
    target_window_size = data_cnf['target_window_size']
    genome_window_step = data_cnf['genome_window_step']

    genome_sizes_file = data_cnf['genome_sizes_file']
    blacklist_file = data_cnf['blacklist_file']

    result_filefolder = input_dir + "model_data/"
    valid_chroms = data_cnf['valid_chroms']
    test_chroms = data_cnf['test_chroms']

    Path(result_filefolder).mkdir(parents=True, exist_ok=True)
    print(f'make sure {result_filefolder} contain no existing file ')


    pybedtools.set_tempdir('/Users/cmf/Downloads/tmp')

    blacklist = make_blacklist(blacklist_file, genome_sizes_file, genome_window_size)
    tfs, positive_windows_list, nonnegative_regions_list = load_chip_multiTask(input_dir,genome_sizes_file, target_window_size, genome_window_size, genome_window_step, blacklist)
    
    for positive_windows, nonnegative_regions, tf in zip(positive_windows_list, nonnegative_regions_list, tfs):
        make_features_SingleTask(genome_sizes_file, tf, positive_windows, target_window_size, genome_window_size, nonnegative_regions, valid_chroms, test_chroms, result_filefolder)


if __name__ == '__main__':
    main()