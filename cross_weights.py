import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', help='Process', default='train')
parser.add_argument('-data', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-b', '--batch', help='Batch size', default=16, type=int)
parser.add_argument('-c', '--classes', help='Number of classes', default=101, type=int)
parser.add_argument('-e', '--epoch', help='Number of epochs', default=5, type=int)
parser.add_argument('-dropout', '--dropout', help='Dropout', default=0.8, type=float)
parser.add_argument('-r', '--retrain', help='Number of old epochs when retrain', default=0, type=int)
parser.add_argument('-cross', '--cross', help='Cross fold', default=1, type=int)
parser.add_argument('-s', '--summary', help='Show model', default=0, type=int)
parser.add_argument('-lr', '--lr', help='Learning rate', default=1e-3, type=float)
parser.add_argument('-decay', '--decay', help='Decay', default=0.0, type=float)
parser.add_argument('-fine', '--fine', help='Fine-tuning', default=1, type=int)
parser.add_argument('-n', '--neural', help='LSTM neural', default=256, type=int)
args = parser.parse_args()
print (args)

import sys
import config
import models
from keras import optimizers

process = args.process
old_epochs = 0
batch_size = args.batch
classes = args.classes
epochs = args.epoch
cross_index = args.cross
dataset = args.dataset

seq_len = 3
n_neurons = args.neural
dropout = args.dropout
pre_file = 'inception_spatial2fc_{}'.format(n_neurons)

model = models.InceptionSpatial(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=None, dropout=dropout, fine=1, retrain=False,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)
    
model.load_weights('weights/{}_{}e_cr{}.h5'.format(pre_file,epochs,cross_index))
weights_0 = model.layers[0].get_weights()
print(weights_0)
    
