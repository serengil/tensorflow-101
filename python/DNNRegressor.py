import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils #export_savedmodel

import matplotlib.pyplot as plt
import io

#----------------------------

input = np.loadtxt("sine.csv", dtype='f', delimiter=',')
#print(input)

row = input.shape[0]
col = input.shape[1]

#----------------------------
#attributes and labels
attributes = [[0 for i in range(col-1)] for j in range(row)]
labels = []
for i in range(row):
	labels.append(0)

for i in range(0, row):
	for j in range(0, col):
		if j < col -1:
			attributes[i][j] = input[i][j]
		else:
			labels[i] = input[i][j]

data =np.array(attributes, 'float32')
target = np.array(labels, 'float32')

#----------------------------

#nn learning parameters 
learningRate = 0.1
epoch = 10000

#----------------------------

feature_columns = [tf.contrib.layers.real_valued_column("", dimension = col-1)]

#neural network model
regressor = tf.contrib.learn.DNNRegressor(
	feature_columns = feature_columns
	, hidden_units = [4] #a hidden layer consisting of 4 nodes
	, optimizer = tf.train.GradientDescentOptimizer(learningRate)
	, activation_fn = tf.nn.sigmoid
	, model_dir = "model" #model will be stored in this folder
	, config = tf.contrib.learn.RunConfig(save_checkpoints_secs = 1)
)

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(data, target, every_n_steps = 1000)

regressor.fit(data, target, steps = epoch
	, monitors = [validation_monitor]
)

"""
#this block provides to export nn model language neutrally. in this way, same model can be used in high level systems such as Java
feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)
serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
regressor.export_savedmodel(regressor.model_dir+"\export", serving_input_fn, as_text=True)
"""

def test_set():
	return np.array(attributes, np.float32)

predictions = regressor.predict_scores(input_fn = test_set)

#--------------------------
#dumping predictions and actual sets

index = 0
for i in predictions:
	print("actual: ", target[index],", predic: ", i)
	index = index + 1

#--------------------------

#model performance
eva = regressor.evaluate(data, target)
print("MSE: ", eva["loss"])

#--------------------------

#visualizing predictions and actuals in TensorBoard

#predictions have to be restored for processing
predictions = regressor.predict_scores(input_fn = test_set)

actuals = labels
forecasts = list(predictions)

forecast_writer = tf.summary.FileWriter('model/forecast')
actual_writer = tf.summary.FileWriter('model/actual')

for i in range(0, row):
	actual_summary = tf.Summary(
		value = [tf.Summary.Value(tag="summary_tag", simple_value= actuals[i])])
	forecast_summary = tf.Summary(
		value = [tf.Summary.Value(tag="summary_tag", simple_value= forecasts[i])])
		
	actual_writer.add_summary(actual_summary, i)
	forecast_writer.add_summary(forecast_summary, i)

#--------------------------

"""
#x-axis: time, y-axis:value. actual and forecast values are plotted as 2 line in same graph. 
#this graph cannot be readable for displaying too many point
plt.plot(actuals)
plt.plot(forecasts)
plt.show()
"""

"""
#x-axis represents predicted values whereas y-axis represents actual values. 
plt.scatter(actuals, forecasts)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
"""

#----------------------------------
#embedding matplotlib graph in TensorBoard

def create_plot(actuals, forecasts):
	plt.figure()
	plt.scatter(actuals, forecasts)
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	buf.seek(0)
	return buf

plot_buf = create_plot(actuals, forecasts)
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
image = tf.expand_dims(image, 0)

summary_img = tf.summary.image("scatter_plot", image)

sess = tf.Session()

summary = sess.run(summary_img)
writer = tf.summary.FileWriter('model/logs')
writer.add_summary(summary)