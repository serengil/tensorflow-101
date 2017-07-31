import tensorflow as tf
import numpy as np
import pylab as pl

from tensorflow.contrib.factorization.python.ops import clustering_ops
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------
#variables
classes = 3    # define number of clusters
display3D = True

#-----------------------------------------

#dataset
#monthly expenses, net assets
atributes = [
		[1100, 1200]
		, [2200, 2500]
		, [3300, 3600]
		, [2400, 2700]
		, [14100, 3200]
		, [4120, 15200]
		, [3125, 3600]
		, [2400, 13700]
		, [3100, 3200]
		, [4100, 4200]
		, [13100, 13200]
		, [4110, 14200]
		, [5100, 15200]
	 ]

row = len(atributes)
col = len(atributes[0])

print("[", row,"x",col,"] sized input")

if display3D == False:
	for i in range(row):
		pl.scatter(atributes[i][0], atributes[i][1], c='black')
	pl.show()	

#-----------------------------------------

model = tf.contrib.learn.KMeansClustering(
		classes
		, distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE #SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE
		, initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
	)

#-----------------------------------------

def train_input_fn():
    data = tf.constant(atributes, tf.float32)
    return (data, None)

model.fit(input_fn=train_input_fn, steps=5000)

print("--------------------")
print("kmeans model: ",model)

def predict_input_fn():
	return np.array(atributes, np.float32)

predictions = model.predict(input_fn=predict_input_fn, as_iterable=True)

colors = ['orange', 'red', 'blue']

print("--------------------")

if display3D == True:
	fig = pl.figure()
	ax = fig.add_subplot(111, projection='3d')

index = 0
for i in predictions:	
	print("[", atributes[index],"] -> cluster_",i['cluster_idx'])
	
	if display3D == False:
		pl.scatter(atributes[index][0], atributes[index][1], c=colors[i['cluster_idx']]) #2d graph
	if display3D == True:
		ax.scatter(atributes[index][0], atributes[index][1], c=colors[i['cluster_idx']]) #3d graph
	
	index  = index + 1

pl.show()

#-----------------------------------------

"""
#to predict the cluster of new instances
testset = [[1.3, 1.2]
		, [2.1, 2.3]
	 ]
def newinstances_input_fn():
	return np.array(testset, np.float32)
predictions = model.predict(input_fn=newinstances_input_fn, as_iterable=True)
for i in predictions:
	print("cluster_",i['cluster_idx'])
"""