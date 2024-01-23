from torchviz import make_dot

from tfnet.data_utils import *
from tfnet.datasets import TFBindDataset
from tfnet.models import Model
from tfnet.networks import TFNet
from torch.utils.data.dataloader import DataLoader
from functools import partial
from tfnet.all_tfs import all_tfs

import matplotlib.pyplot as plt 

conv_num = [16, 16, 16, 16, 16]
conv_size = [5, 9, 13, 17, 21]
conv_off = [8, 6, 4, 2, 0]
linear_size = [32, 16]
full_size = [29696, 256,64]

model = TFNet(conv_num = conv_num, conv_size = conv_size, conv_off = conv_off, linear_size = linear_size, full_size = full_size)

tf_name_seq = get_tf_name_seq('data/tf_pseudosequences.txt')
get_data_fn = partial(get_data, tf_name_seq=tf_name_seq)
train_data = get_data_fn('/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/pos_data_test.txt')
train_loader = DataLoader(TFBindDataset(train_data), batch_size=128, shuffle=True)
test_features, test_labels = next(iter(train_loader))

model_labels = model(test_features[0],test_features[1])

Model_structure = make_dot(model_labels, params=dict(model.named_parameters()))
Model_structure.render("model_graph", format="pdf")