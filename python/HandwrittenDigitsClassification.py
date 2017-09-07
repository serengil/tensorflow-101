#this code classifies handwritten digits 
#also known as MNIST - Modified National Institute of Standards and Technology database

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math

#tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------
#variables
mnist_data = "C:/Users/IS96273/Desktop/tmp/MNIST_data"
model_dir = "C:/Users/IS96273/Desktop/tmp/tfmodels/mnist_tflearn/"

epoch = 500
learningRate = 0.1

MNIST_DATASET = input_data.read_data_sets(mnist_data)

train_data = np.array(MNIST_DATASET.train.images, 'int64')
train_target = np.array(MNIST_DATASET.train.labels, 'int64')
print("training set consists of ", len(MNIST_DATASET.train.images), " instances")

test_data = np.array(MNIST_DATASET.test.images, 'int64')
test_target = np.array(MNIST_DATASET.test.labels, 'int64')
print("test set consists of ", len(MNIST_DATASET.test.images), " instances")

#-----------------------------------------------
#visualization
print("input layer consists of ", len(MNIST_DATASET.train.images[1]), " features ("
	,math.sqrt(len(MNIST_DATASET.train.images[1])), "x", math.sqrt(len(MNIST_DATASET.train.images[1]))," pixel images)") #28x28 = 784 input feature
"""
print("features: ", MNIST_DATASET.train.images[1])
print("labels: ", MNIST_DATASET.train.labels[1])
"""

"""
#to display a sample
sample = 2
#print(MNIST_DATASET.train.images[sample])
print(MNIST_DATASET.train.labels[sample])

X = MNIST_DATASET.train.images[sample]
X = X.reshape([28, 28]);
#X = X.reshape([math.sqrt(len(MNIST_DATASET.train.images[1])), math.sqrt(len(MNIST_DATASET.train.images[1]))]);
plt.gray()
plt.imshow(X)
plt.show()
"""
#-----------------------------------------------
def applyDNNClassifier(num_steps, logdir):
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(MNIST_DATASET.train.images[1]))]
	
	classifier = tf.contrib.learn.DNNClassifier(
		feature_columns=feature_columns,
		n_classes=10, #0 to 9 - 10 classes
		hidden_units=[128, 32],  #2 hidden layers consisting of 128 and 32 units respectively
		optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate),
		model_dir=logdir
	)
	
	#classifier.fit(train_data, train_target, steps=num_steps)
	print("\n---training is over...")
	
	"""
	predictions = classifier.predict_classes(test_data)
	index = 0
	for i in predictions:
		if index < 10:
			print("actual: ", test_target[index], ", prediction: ", i)
			
			pred = MNIST_DATASET.test.images[index]
			pred = pred.reshape([28, 28]);
			plt.gray()
			plt.imshow(pred)
			plt.show()
			
		index  = index + 1
	"""
	
	print("\n---evaluation...")
	accuracy_score = classifier.evaluate(test_data, test_target, steps=100)['accuracy']
	print("accuracy: ", 100*accuracy_score,"%")
	
#---------------------------------------------
#main
applyDNNClassifier(epoch, model_dir)
