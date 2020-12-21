import numpy as np

#-------------------------

import torch

#Network structure: https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py
#Pre-trained weights: https://github.com/clcarwin/sphereface_pytorch/blob/master/model/sphere20a_20171020.7z
PATH = "sphere20a_20171020.pth"

model = torch.load(PATH)
print(type(model))

model_layers = list(model.items())

source = {}
idx = 0
for layer in model_layers:
	layer_name = layer[0]
	layer_weights = layer[1].detach().numpy()
	if idx < len(model_layers) - 1:
		layer_weights = np.transpose(layer_weights)
	
	if 'relu' in layer_name: # and len(layer_weights.shape) == 3 and layer_weights[0] == 1 and layer_weights[1] == 1: 
		layer_weights = np.expand_dims(layer_weights, axis = 0)
		layer_weights = np.expand_dims(layer_weights, axis = 0)
	
	print("storing weights of ",layer_name, layer_weights.shape)
	source[layer_name] = layer_weights
	idx = idx + 1


#summary
"""
#source model
conv1_1.weight (64, 3, 3, 3)
conv1_1.bias (64,)
relu1_1.weight (64,)
conv1_2.weight (64, 64, 3, 3)
conv1_2.bias (64,)
relu1_2.weight (64,)
conv1_3.weight (64, 64, 3, 3)
conv1_3.bias (64,)
relu1_3.weight (64,)
conv2_1.weight (128, 64, 3, 3)
conv2_1.bias (128,)
relu2_1.weight (128,)
conv2_2.weight (128, 128, 3, 3)
conv2_2.bias (128,)
relu2_2.weight (128,)
conv2_3.weight (128, 128, 3, 3)
conv2_3.bias (128,)
relu2_3.weight (128,)
conv2_4.weight (128, 128, 3, 3)
conv2_4.bias (128,)
relu2_4.weight (128,)
conv2_5.weight (128, 128, 3, 3)
conv2_5.bias (128,)
relu2_5.weight (128,)
conv3_1.weight (256, 128, 3, 3)
conv3_1.bias (256,)
relu3_1.weight (256,)
conv3_2.weight (256, 256, 3, 3)
conv3_2.bias (256,)
relu3_2.weight (256,)
conv3_3.weight (256, 256, 3, 3)
conv3_3.bias (256,)
relu3_3.weight (256,)
conv3_4.weight (256, 256, 3, 3)
conv3_4.bias (256,)
relu3_4.weight (256,)
conv3_5.weight (256, 256, 3, 3)
conv3_5.bias (256,)
relu3_5.weight (256,)
conv3_6.weight (256, 256, 3, 3)
conv3_6.bias (256,)
relu3_6.weight (256,)
conv3_7.weight (256, 256, 3, 3)
conv3_7.bias (256,)
relu3_7.weight (256,)
conv3_8.weight (256, 256, 3, 3)
conv3_8.bias (256,)
relu3_8.weight (256,)
conv3_9.weight (256, 256, 3, 3)
conv3_9.bias (256,)
relu3_9.weight (256,)
conv4_1.weight (512, 256, 3, 3)
conv4_1.bias (512,)
relu4_1.weight (512,)
conv4_2.weight (512, 512, 3, 3)
conv4_2.bias (512,)
relu4_2.weight (512,)
conv4_3.weight (512, 512, 3, 3)
conv4_3.bias (512,)
relu4_3.weight (512,)
fc5.weight (512, 21504)
fc5.bias (512,)
fc6.weight (512, 10574)
"""
#-------------------------

import tensorflow
from tensorflow.python.keras.engine import training
from tensorflow import keras
from deepface.basemodels import ArcFace

inputs = tensorflow.keras.layers.Input(shape=(96, 112, 3)) #input = B*3*112*96
x = ArcFace.stack1(inputs, 64, 3, name='conv1', use_bias = True)
x = ArcFace.stack1(x, 128, 5, name='conv2', use_bias = True)
x = ArcFace.stack1(x, 256, 9, name='conv3', use_bias = True)
x = ArcFace.stack1(x, 512, 3, name='conv4', use_bias = True)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation='linear', use_bias=True, name='fc5')(x)

#there are 454590 images of 10574 identites in CASIA database
#x = keras.layers.Dense(10574, activation=None, use_bias=False, name='fc6')(x)

#model = training.Model(inputs, x, name='SphereFace')
model = keras.models.Model(inputs, x, name='SphereFace')

#-----------------------------------------------------
#transfer weights

print(type(model))

for layer in model.layers:
	
	#if True:
	if '1_conv' in layer.name or 'relu' in layer.name or 'fc' in layer.name:
		weights = layer.get_weights()
		layer_idx = 0
		
		weight_content = []
		for weight in weights:
			
			duty = 'weight' if layer_idx == 0 else 'bias'
			
			layer_name = '%s.%s' % (layer.name.replace("block", "").replace("_1_conv", ""), duty)
			
			if 'prelu' in layer_name:
				layer_name = layer_name.replace("_1_prelu", "").replace("conv", "relu")
			
			#print(layer_name, weight.shape) #,layer.output_shape
			
			print("retrieving weights of ", layer_name)
			weight_content.append(source[layer_name])
			
			layer_idx = layer_idx + 1
		
		model.get_layer(layer.name).set_weights(weight_content)
		print(layer.name," is updated")

model.save_weights("sphereface_weights.h5")

"""
for layer in model.layers:
	print(layer.name,": ",layer.output_shape)
"""

#-----------------------------------------------------
#model.load_weights("sphereface_weights.h5")
print("SphereFace restored")

w, h = model.layers[0].input_shape[1:3]
print("input shape is ", w, h)

#-----------------------------------------------------
from deepface.commons import functions
from deepface.commons import distance as dst

def verify(img1_path, img2_path):
	
	backend = 'opencv'

	img1 = functions.preprocess_face(img1_path, target_size = (h, w), detector_backend = backend)
	img2 = functions.preprocess_face(img2_path, target_size = (h, w), detector_backend = backend)
	
	#-----------------------------------------------------
	
	img1_embedding = model.predict(img1)[0]
	img2_embedding = model.predict(img2)[0]
	
	#-----------------------------------------------------
	#we might need to change this logic: http://cseweb.ucsd.edu/~mkchandraker/classes/CSE252C/Spring2020/Homeworks/hw2-CSE252C-Sol.html
	
	#print(np.argmax(img1_embedding), np.argmax(img2_embedding))
	
	return (dst.findEuclideanDistance(img1_embedding, img2_embedding)
			, dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))
			, dst.findCosineDistance(img1_embedding, img2_embedding)
			)

#----------------------------------------------------------

import pandas as pd

df = pd.read_csv("master.csv")

distances = []
for index, instance in df.iterrows():
	print(index," / ",df.shape[0])
	distance = []
	decision = instance["Decision"]
	img1_path = "../deepface/tests/dataset/"+instance["file_x"]
	img2_path = "../deepface/tests/dataset/"+instance["file_y"]
	
	try:		
		euclidean, euclidean_l2, cosine = verify(img1_path, img2_path)
		
		distance.append(decision)
		distance.append(cosine)
		distance.append(euclidean)
		distance.append(euclidean_l2)
		distances.append(distance)
	except Exception as err:
		print(str(err))
	
pivot = pd.DataFrame(distances, columns = ["target", "cosine", "euclidean", "euclidean_l2"])
print(pivot.head())

pivot.to_csv('scores.csv', index=False)