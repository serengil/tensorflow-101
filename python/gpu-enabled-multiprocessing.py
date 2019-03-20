import pandas as pd
import multiprocessing
from multiprocessing import Pool

def train(index, df):
	import tensorflow as tf
	import keras
	from keras.models import Sequential
	
	#------------------------------
	#this block enables GPU enabled multiprocessing
	core_config = tf.ConfigProto()
	core_config.gpu_options.allow_growth = True
	session = tf.Session(config=core_config)
	keras.backend.set_session(session)
	#------------------------------
	#prepare input and output values
	df = df.drop(columns=['index'])
	
	data = df.drop(columns=['target']).values
	target = df['target']
	#------------------------------
	model = Sequential()
	model.add(Dense(5 #num of hidden units
	, input_shape=(data.shape[1],))) #num of features in input layer
	model.add(Activation('sigmoid'))
	
	model.add(Dense(1))#number of nodes in output layer
	model.add(Activation('sigmoid'))
	
	model.compile(loss='mse', optimizer=keras.optimizers.Adam())
	#------------------------------
	model.fit(data, target, epochs = 5000, verbose = 1)
	model.save("model_for_%s.hdf5" % index)
	#------------------------------
	#finally, close sessions
	session.close()
	keras.backend.clear_session() 

#-----------------------------
#main program

multiprocessing.set_start_method('spawn', force=True)

df = pd.read_csv("dataset.csv")

my_tuple = [(i, df[df['index'] == i]) for i in range(0, 20)]

with Pool(10) as pool: 
	pool.starmap(train, my_tuple)
