import numpy as np
import sys
import config
import models
import keras as K
from keras import optimizers
from keras.applications import InceptionV3
from inceptionv3 import *

inception = InceptionV3(input_shape=(299,299,3), pooling='avg', include_top=False, weights='imagenet',)

inceptionv3a = Inception_v3a(input_shape=(299,299,3))
inceptionv3b =Inception_v3b(inceptionv3a.output_shape, inceptionv3a.output)
inceptionv3c = Inception_v3c(inceptionv3b.output_shape, inceptionv3b.output)

for i in range(len(inceptionv3a.layers)):
	inceptionv3a.layers[i].set_weights(inception.layers[i].get_weights())
inceptionv3a.save_weights('./data/InceptionV3a.h5')

for i in range(len(inceptionv3b.layers)):
	inceptionv3b.layers[i].set_weights(inception.layers[i+len(inceptionv3a.layers)].get_weights())
inceptionv3a.save_weights('./data/InceptionV3b.h5')

for i in range(len(inceptionv3c.layers)):
	inceptionv3c.layers[i].set_weights(inception.layers[i+len(inceptionv3b.layers)+len(inceptionv3c.layers)].get_weights())
inceptionv3a.save_weights('./data/InceptionV3c.h5')
