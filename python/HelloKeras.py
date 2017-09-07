import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

#----------------------------
#preparing data for Exclusive OR (XOR)

attributes = [
	#x1, x2
	[0 ,0]
	, [0, 1]
	, [1, 0]
	, [1, 1]
]

labels = [
	#is_0, is_1 -> only a column can be 1 in labels variable
	[1, 0] 
	, [0, 1]
	, [0, 1]
	, [1, 0]
]

#transforming attributes and labels matrixes to numpy
data = np.array(attributes, 'int64')
target = np.array(labels, 'int64')

#----------------------------
#creating model

model = Sequential()
model.add(Dense(3 #num of hidden units
	, input_shape=(len(attributes[0]),))) #num of features in input layer
model.add(Activation('sigmoid')) #activation function from input layer to 1st hidden layer
model.add(Dense(len(labels[0]))) #num of classes in output layer
model.add(Activation('softmax')) #activation function from 1st hidden layer to output layer

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam')

#training
score = model.fit(data, target, epochs=100, verbose=0)

print(score.history)