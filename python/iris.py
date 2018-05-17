import tensorflow as tf
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
#----------------------------
attributes = np.genfromtxt("iris-attr.data", delimiter=",")
label = np.genfromtxt("iris-labels.data", dtype="int64")

num_classes = 3
label = keras.utils.to_categorical(label, num_classes)
#----------------------------
#creating model

model = Sequential()
model.add(Dense(4 #num of hidden units
	, input_shape=(len(attributes[0]),))) #num of features in input layer
model.add(Activation('sigmoid')) #activation function from input layer to 1st hidden layer
model.add(Dense(len(label[0]))) #num of classes in output layer
model.add(Activation('sigmoid')) #activation function from 1st hidden layer to output layer

#----------------------------
#compile
model.compile(loss='categorical_crossentropy', optimizer='adam')

#training
model.fit(attributes, label, epochs=1000, verbose=0)

predictions = model.predict(attributes)

index = 0
for i in predictions:
	#print(np.argmax(i)," (",i,")")
	
	pred = np.argmax(i)
	actual = np.argmax(label[index])
	print(" prediction: ",pred," - actual: ",actual)
	index = index + 1