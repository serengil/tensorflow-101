#This code classifies handwritten digits 
#Also known as MNIST - Modified National Institute of Standards and Technology database

#This configuration produced 98.01% accuracy for test set whereas it produced 99.77% accuracy for trainset. 
#Producing close accuracy rates is expected for re-run (random initialization causes to produce different results each time)

#blog post: https://sefiks.com/2017/09/11/handwritten-digit-classification-with-tensorflow/

#-----------------------------------------------

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math

tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------
#variables

epoch = 15000
learningRate = 0.1
batch_size = 120

mnist_data = "C:/tmp/MNIST_data"

trainForRandomSet = True

#-----------------------------------------------
#data process and transformation

MNIST_DATASET = input_data.read_data_sets(mnist_data)

train_data = np.array(MNIST_DATASET.train.images, 'float32')
train_target = np.array(MNIST_DATASET.train.labels, 'int64')
print("training set consists of ", len(MNIST_DATASET.train.images), " instances")

test_data = np.array(MNIST_DATASET.test.images, 'float32')
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
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(MNIST_DATASET.train.images[1]))]

classifier = tf.contrib.learn.DNNClassifier(
	feature_columns=feature_columns
	, n_classes=10 #0 to 9 - 10 classes
	, hidden_units=[128, 32]  #2 hidden layers consisting of 128 and 32 units respectively
	, optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate)
	, activation_fn = tf.nn.relu
	#, activation_fn = tf.nn.softmax
	, model_dir="model"
)

#----------------------------------------
#training

if trainForRandomSet == False:
	#train on all trainset
	classifier.fit(train_data, train_target, steps=epoch)
else:
	def generate_input_fn(data, label):	
		image_batch, label_batch = tf.train.shuffle_batch(
			[data, label]
			, batch_size=batch_size
			, capacity=8*batch_size
			, min_after_dequeue=4*batch_size
			, enqueue_many=True
		)
		return image_batch, label_batch
	
	def input_fn_for_train():
		return generate_input_fn(train_data, train_target)
	
	#train on small random selected dataset
	classifier.fit(input_fn=input_fn_for_train, steps=epoch)

print("\n---training is over...")

#----------------------------------------
#apply to make predictions

predictions = classifier.predict_classes(test_data)
index = 0
for i in predictions:
	if index < 10: #visualize first 10 items on test set
		print("actual: ", test_target[index], ", prediction: ", i)
		
		pred = MNIST_DATASET.test.images[index]
		pred = pred.reshape([28, 28]);
		plt.gray()
		plt.imshow(pred)
		plt.show()
		
	index  = index + 1

#----------------------------------------
#calculationg overall accuracy

print("\n---evaluation...")
accuracy_score = classifier.evaluate(test_data, test_target, steps=epoch)['accuracy']
print("accuracy: ", 100*accuracy_score,"%")

