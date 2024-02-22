#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : main.py
@Time : 2023/11/09 11:18:29
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import click
import numpy as np
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from logzero import logger

from tfnet.data_utils import *
from tfnet.datasets import TFBindDataset
from tfnet.models_epoch import Model
from tfnet.networks_modify import TFNet
from tfnet.evaluation import output_eval, output_predict, CUTOFF
from tfnet.all_tfs import all_tfs

import pdb


# code
def train(model, data_cnf, model_cnf, train_data, valid_data=None, class_weights_dict = None, random_state=1240):
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 0.2),
                                                  random_state=random_state)
        
    model.train(data_cnf, model_cnf, train_data, valid_data, class_weights_dict, **model_cnf['train']) # for samples_per_epoch

    logger.info(f'Finish training model {model.model_path}')


def test(model, data_cnf, model_cnf, test_data):
    data_loader = DataLoader(TFBindDataset(test_data, data_cnf['genome_fasta_file'], data_cnf['mappability'], data_cnf['chromatin'], **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'])
    return model.predict(data_loader)


def generate_cv_id(length, num_groups=5):
    base_size = length // num_groups
    extra_size = length % num_groups
    group_sizes = [base_size + 1 if i < extra_size else base_size for i in range(num_groups)]
    #labels = np.repeat(np.arange(1, num_groups + 1), group_sizes)
    labels = np.concatenate([np.full(size, i) for i, size in enumerate(group_sizes)])
    return labels


def get_binding_core(data_list, model_cnf, model_path, start_id, num_models, core_len=9):
    scores_list = []
    for model_id in range(start_id, start_id + num_models):
        model = Model(TFNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), pooling=False,
                      **model_cnf['model'])
        scores_list.append(test(model, model_cnf, data_list))
    return (scores:=np.mean(scores_list, axis=0)).argmax(-1), scores


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(('train', 'eval', 'predict','5cv', 'loo', 'lomo')), default=None)
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=1)
@click.option('-c', '--continue', 'continue_train', is_flag=True)
@click.option('-a', '--allele', default=None)
def main(data_cnf, model_cnf, mode, continue_train, start_id, num_models, allele):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    logger.info(f'Model Name: {model_name}')
    model_path = Path(model_cnf['path'])/f'{model_name}.pt'
    res_path = Path(data_cnf['results'])/f'{model_name}'
    Path(data_cnf['results']).mkdir(parents=True, exist_ok=True)
    model_cnf.setdefault('ensemble', 20)
    tf_name_seq = get_tf_name_seq(data_cnf['tf_seq'])

    seq_name_tf = {v : k for k ,v in tf_name_seq.items()}

    get_data_fn = partial(get_data_lazy, tf_name_seq=tf_name_seq, genome_fasta_file= data_cnf['genome_fasta_file'], DNA_N = model_cnf['padding']['DNA_N'])

    classweights = model_cnf['classweights']

    if classweights:
        class_weights_dict = calculate_class_weights_dict(data_cnf['train'])
    else :
        class_weights_dict = None


    if mode == "train":
        train_data = get_data_fn(data_cnf['train']) if mode is None or mode == 'train' else None
        valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None
        for model_id in range(start_id, start_id + num_models):
            model = Model(TFNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict, tf_len = model_cnf['padding']['tf_len'],
                          **model_cnf['model'])
            if not continue_train or not model.model_path.exists():
                train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data, class_weights_dict = class_weights_dict)
            
    elif mode == 'eval':
        test_data = get_data_fn(data_cnf['test'])
        shift = int((model_cnf['padding']['DNA_len'] - model_cnf['padding']['target_len'])/2)
        
        #chr, start, stop, targets_lists = [x[0] for x in test_data], [x[1] + shift for x in test_data], [x[2] - shift for x in test_data], [x[-2] for x in test_data] # depend on the input data len
        chr, start, stop, targets_lists, tfs, celltypes = [x[0] for x in test_data], [x[1] for x in test_data], [x[2] for x in test_data], [x[3] for x in test_data], [x[4] for x in test_data],  [x[5].rstrip('\n') for x in test_data]
        tf_names = [ seq_name_tf[i] for i in tfs]
        scores_lists = []
        for model_id in range(start_id, start_id + num_models):
            model = Model(TFNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict, tf_len = model_cnf['padding']['tf_len'],
                          **model_cnf['model'])
            scores_lists.append(test(model, data_cnf, model_cnf, test_data=test_data))
        output_eval(chr, start, stop, np.array(targets_lists), np.mean(scores_lists, axis=0), tf_names, celltypes, res_path)
    
    elif mode == 'predict':
        predict_data = get_data_fn(data_cnf['predict'])
        shift = int((model_cnf['padding']['DNA_len'] - model_cnf['padding']['target_len'])/2)

        chr, start, stop, targets_lists, tfs, celltypes = [x[0] for x in predict_data], [x[1] for x in predict_data], [x[2] for x in predict_data], [x[3] for x in predict_data], [x[4] for x in predict_data],  [x[5].rstrip('\n') for x in predict_data]
        tf_names = [ seq_name_tf[i] for i in tfs]
        scores_lists = []
        for model_id in range(start_id, start_id + num_models):
            model = Model(TFNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict, tf_len = model_cnf['padding']['tf_len'],
                          **model_cnf['model'])
            scores_lists.append(test(model, data_cnf, model_cnf, test_data=predict_data))
        output_predict(chr, start, stop, np.mean(scores_lists, axis=0), tf_names, celltypes, res_path)

    elif mode == '5cv':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        data_group_name, atac_signal, data_truth = [x[0] for x in data], [x[1] for x in data], [x[2] for x in data]
        cv_id_len = data.shape[0]
        # ---------------------- generate cv id for use rather than read a input ---------------------- #
        cv_id = generate_cv_id(cv_id_len)
        assert len(data) == len(cv_id)
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            scores_ = np.empty(len(data)*len(all_tfs), dtype=np.float32).reshape(len(data), len(all_tfs))
            for cv_ in range(5):
                
                train_data, test_data = data[cv_id != cv_], data[cv_id == cv_]
                model = Model(TFNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}'), class_weights_dict = class_weights_dict, tf_len = model_cnf['padding']['tf_len'],
                              **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, class_weights_dict = class_weights_dict)
                scores_[cv_id == cv_] = test(model, model_cnf, test_data=test_data)

                scores_list.append(scores_)
                #pdb.set_trace()

                #output_res(np.array(data_group_name)[cv_id == cv_], np.array(data_truth)[cv_id == cv_], np.mean(scores_[cv_id == cv_], axis=0),
                output_eval(np.array(data_group_name)[cv_id == cv_], np.array(data_truth)[cv_id == cv_], scores_[cv_id == cv_],          
                       res_path.with_name(f'{res_path.stem}-5CV'))


    elif mode == 'loo' or mode == 'lomo':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            group_names, group_names_, truth_, scores_ = np.asarray([x[0] for x in data]), [], [], []
            for name_ in sorted(set(group_names)):
                train_data, train_cv_id = data[group_names != name_], cv_id[group_names != name_]
                test_data, test_cv_id = data[group_names == name_], cv_id[group_names == name_]
                if len(test_data) > 30 and len([x[-1] for x in test_data if x[-1] >= CUTOFF]) >= 3:
                    for cv_ in range(5):
                        model = Model(TFNet,
                                      model_path=model_path.with_stem(F'{model_path.stem}-{name_}-{model_id}-CV{cv_}'), class_weights_dict = class_weights_dict, tf_len = model_cnf['padding']['tf_len'],
                                      **model_cnf['model'])
                        if not model.model_path.exists() or not continue_train:
                            train(model, data_cnf, model_cnf, train_data[train_cv_id != cv_], class_weights_dict=class_weights_dict)
                        test_data_ = test_data[test_cv_id == cv_]
                        group_names_ += [x[0] for x in test_data_]
                        truth_ += [x[-1] for x in test_data_]
                        scores_ += test(model, model_cnf, test_data_).tolist()
            scores_list.append(scores_)
            output_eval(group_names_, truth_, np.mean(scores_list, axis=0), res_path.with_name(f'{res_path.stem}-LOMO'))


if __name__ == '__main__':
    main()
