import tensorflow as tf
import numpy as np
import logging

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

tf.logging.set_verbosity(tf.logging.INFO)

atributes = [
	[0, 0]
	, [0, 1]
	, [1, 0]
	, [1, 1]
]

labels = [
	0
	, 1
	, 1
	, 0
]

data = np.array(atributes, 'int64')
target = np.array(labels, 'int64')

feature_columns = [tf.contrib.layers.real_valued_column(""
							, dimension=len(atributes[0]) #attributes consist of two columns: x1 and x2.
							, dtype=tf.float32)]

learningRate = 0.1
epoch = 2000

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(data, target, every_n_steps = 500)

sigmoid_classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns
	, hidden_units = [3]
	, activation_fn = tf.nn.sigmoid
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, model_dir = "model/sigmoid"
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 10)
)

tanh_classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns
	, hidden_units = [3]
	, activation_fn = tf.nn.tanh
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, model_dir = "model/tanh"
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 10)
)

softplus_classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns
	, hidden_units = [3]
	, activation_fn = tf.nn.softplus
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, model_dir = "model/softplus"
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 10)
)

relu_classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns
	, hidden_units = [3]
	, activation_fn = tf.nn.relu
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, model_dir = "model/relu"
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 10)
)

sigmoid_classifier.fit(data, target, steps = epoch, monitors = [validation_monitor])
tanh_classifier.fit(data, target, steps = epoch, monitors = [validation_monitor])
softplus_classifier.fit(data, target, steps = epoch, monitors = [validation_monitor])
relu_classifier.fit(data, target, steps = epoch, monitors = [validation_monitor])