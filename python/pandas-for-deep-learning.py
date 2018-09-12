import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
#--------------------------------

num_classes = 3 #Iris-setosa,Iris-versicolor,Iris-virginica

#--------------------------------
def createNetwork():
	model = Sequential()

	model.add(Dense(4 #num of hidden units
		, input_shape=(4,))) #num of features in input layer
	model.add(Activation('sigmoid')) #activation function from input layer to 1st hidden layer
	
	model.add(Dense(num_classes)) #num of classes in output layer
	model.add(Activation('sigmoid')) #activation function from 1st hidden layer to output layer
	
	return model

model = createNetwork()
model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam(lr=0.007)
    , metrics=['accuracy']
)
#--------------------------------
chunk_size = 30
epochs = 1000

for epoch in range(0, epochs): #epoch should be handled here, not in fit command!
	
	if epoch % 100 == 0:
		print("epoch ",epoch)
	
	chunk_index = 0
	for chunk in pd.read_csv("iris.data", chunksize=chunk_size
		, names = ["sepal_length","sepal_width","petal_length","petal_width","class"]):
		
		#print("current chunk: ",chunk_index*chunk_size)
				
		current_set = chunk.values #convert df to numpy array
				
		features = current_set[:,0:4]
		labels = current_set[:,4]
		
		for i in range(0,labels.shape[0]):
			if labels[i] == 'Iris-setosa':
				labels[i] = 0
			elif labels[i] == 'Iris-versicolor':
				labels[i] = 1
			elif labels[i] == 'Iris-virginica':
				labels[i] = 2
		
		labels = keras.utils.to_categorical(labels, num_classes)
		
		#------------------------------------
		model.fit(features, labels, epochs=1, verbose=0) #epochs handled in the for loop above
		
		chunk_index = chunk_index + 1
#-------------------------------------------
df = pd.read_csv("iris.data", names = ["sepal_length","sepal_width","petal_length","petal_width","class"])

for index, row in df.iterrows():
	features = row.values[0:4]
	actual_label = row.values[4]
		
	prediction = model.predict(np.array([features]))
	prediction = np.argmax(prediction)
	
	if prediction == 0:
		predicted_class = "Iris-setosa"
	elif prediction == 1:
		predicted_class = "Iris-versicolor"
	elif prediction == 2:
		predicted_class = "Iris-virginica"
	
	if predicted_class != actual_label:
		print("*", end='')
		
	print(" prediction: ",predicted_class, " - actual: ",actual_label)