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

#data = np.array(atributes, 'float32') #data and target tranformed to int instead of float, because exception throws for metric operations
#target = np.array(labels, 'float32')
data = np.array(atributes, 'int64')
target = np.array(labels, 'int64')

feature_columns = [tf.contrib.layers.real_valued_column(""
							, dimension=len(atributes[0]) #attributes consist of two columns: x1 and x2.
							, dtype=tf.float32)]

learningRate = 0.1
epoch = 10000

#available metrics: https://www.tensorflow.org/api_docs/python/tf/metrics
validation_metrics = {
	"accuracy": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_accuracy
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "precision": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_precision
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "recall": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_recall
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "mean_absolute_error": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_mean_absolute_error
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "false_negatives": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_false_negatives
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "false_positives": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_false_positives
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
	, "true_positives": tf.contrib.learn.MetricSpec(metric_fn = tf.contrib.metrics.streaming_true_positives
		, prediction_key = tf.contrib.learn.PredictionKey.CLASSES)
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(data, target, every_n_steps = 500
	, metrics = validation_metrics #normally tp, fp, fn are not traced. we can add these metrics by adding metrics param
)

classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns
	, hidden_units = [3]
	, activation_fn = tf.nn.sigmoid
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, model_dir = "model"
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 1)
)

classifier.fit(data, target, steps = epoch
	, monitors = [validation_monitor])

#print("params: ", classifier.get_variable_names())
print("total epoch: ", classifier.get_variable_value("global_step"))
print("weights from input layer to hidden layer\n", classifier.get_variable_value("dnn/hiddenlayer_0/weights"))
print("weights from hidden layer to output layer\n", classifier.get_variable_value("dnn/logits/weights"))

"""
#this block is deactivated because I would not exported saved model in external system like Java anymore
feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)
serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
classifier.export_savedmodel(classifier.model_dir+"\export", serving_input_fn, as_text=True)
"""

def test_set():
	return np.array(atributes, np.float32)

predictions = classifier.predict_classes(input_fn = test_set)

#dumping predictions
index = 0
for i in predictions:
	
	print(data[index], " -> actual: ", target[index], ", predict: ", i)
	index  = index + 1

#--------------------------------

#dumping metrics
success_metrics = classifier.evaluate(data, target, metrics = validation_metrics)
print("FN: ", success_metrics["false_negatives"])
print("FP: ", success_metrics["false_positives"])
print("TP: ", success_metrics["true_positives"])
print("-----------------")
print("precision: ", success_metrics["precision"]) # TP / (FP + TP)
print("recall: ", success_metrics["recall"]) # TP / (FN + TP)
print("accuracy: ", success_metrics["accuracy"])
print("mae: ", success_metrics["mean_absolute_error"])
