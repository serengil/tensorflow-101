import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math

tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------
#variables

epoch = 2000
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
print("input layer consists of ", len(MNIST_DATASET.train.images[1])," features")

#-----------------------------------------------
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(MNIST_DATASET.train.images[1]))]

classifier = tf.contrib.learn.DNNClassifier(
	feature_columns=feature_columns
	, n_classes=10 #0 to 9 - 10 classes
	, hidden_units=[128, 64, 32, 16]  #4 hidden layers consisting of 128, 64, 32, 16 units respectively
	#, optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate)
	, optimizer=tf.train.GradientDescentOptimizer(learning_rate=learningRate)
	, activation_fn = tf.nn.sigmoid #activate this to see vanishing gradient
	#, activation_fn = tf.nn.relu #activate this to solve gradient vanishing problem
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
#calculationg overall accuracy

accuracy_score = classifier.evaluate(test_data, test_target, steps=epoch)['accuracy']
print("\n---evaluation...")
print("accuracy: ", 100*accuracy_score,"%")
