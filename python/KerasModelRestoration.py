import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

#----------------------------

train = False
load_all_model = True #if train is False

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

if train == True:	
	model = Sequential()
	model.add(Dense(3 #num of hidden units
		, input_shape=(len(attributes[0]),))) #num of features in input layer
	model.add(Activation('sigmoid')) #activation function from input layer to 1st hidden layer
	model.add(Dense(len(labels[0]))) #num of classes in output layer
	model.add(Activation('softmax')) #activation function from 1st hidden layer to output layer
	
	model_config = model.to_json()
	open("model_structure.json", "w").write(model_config)
	
	#compile
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	#training
	model.fit(data, target, epochs=2000, verbose=0)
	
	model.save("model.hdf5")
	model.save_weights('model_weights.h5')
	
else:
	if load_all_model == True:
		model = load_model("model.hdf5") #model structure, weights
		print("network structure and weights loaded")
	else:
		model = model_from_json(open("model_structure.json", "r").read()) #load structure
		print("network structure loaded")
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		model.load_weights('model_weights.h5') #load weights
		print("weights loaded")

score = model.evaluate(data, target)

print(score)