#Face Recognition with OpenFace
#Author Sefik Ilkin Serengil (sefiks.com)

#You can find the documentation of this code from the following link: 
#https://sefiks.com/

#tested for TensorFlow 1.9.0, Keras 2.2.0 and Python 3.5.5

#-----------------------
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers.merge import Concatenate
from keras import backend as K

from os import listdir
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#-----------------------

dump = False

color = (67,67,67)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(96, 96))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

#------------------------

def builtModel():
	myInput = Input(shape=(96, 96, 3))
	
	x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)
	x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_1')(x)
	x = Conv2D(64, (1, 1), name='conv2')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2D(192, (3, 3), name='conv3')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
	x = Activation('relu')(x)
	Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_2')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)
	
	# Inception3a
	inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
	inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
	inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
	
	inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
	inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
	inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
	
	inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
	inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
	inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
	inception_3a_pool = Activation('relu')(inception_3a_pool)
	inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)
	
	inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
	inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
	inception_3a_1x1 = Activation('relu')(inception_3a_1x1)
	
	inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)
	
	# Inception3b
	inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
	inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
	inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
	
	inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
	inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
	inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
	
	inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
	inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
	inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
	inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
	inception_3b_pool = Activation('relu')(inception_3b_pool)
	inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)
	
	inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
	inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
	inception_3b_1x1 = Activation('relu')(inception_3b_1x1)
	
	inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)
	
	# Inception3c
	inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name='inception_3c_3x3_conv1')(inception_3b)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn1')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
	inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
	inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_3c_3x3_conv'+'2')(inception_3c_3x3)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn'+'2')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
	
	inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_3c_5x5_conv1')(inception_3b)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn1')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
	inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
	inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name='inception_3c_5x5_conv'+'2')(inception_3c_5x5)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn'+'2')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
	
	inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
	inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)
	
	inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)
	
	#inception 4a
	inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_4a_3x3_conv'+'1')(inception_3c)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'1')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
	inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
	inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name='inception_4a_3x3_conv'+'2')(inception_4a_3x3)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'2')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
	
	inception_4a_5x5 = Conv2D(32, (1,1), strides=(1,1), name='inception_4a_5x5_conv1')(inception_3c)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn1')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
	inception_4a_5x5 = ZeroPadding2D(padding=(2,2))(inception_4a_5x5)
	inception_4a_5x5 = Conv2D(64, (5,5), strides=(1,1), name='inception_4a_5x5_conv'+'2')(inception_4a_5x5)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn'+'2')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
	
	inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
	inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
	
	inception_4a_pool = Conv2D(128, (1,1), strides=(1,1), name='inception_4a_pool_conv'+'')(inception_4a_pool)
	inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_pool_bn'+'')(inception_4a_pool)
	inception_4a_pool = Activation('relu')(inception_4a_pool)
	inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)
	
	inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_4a_1x1_conv'+'')(inception_3c)
	inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_1x1_bn'+'')(inception_4a_1x1)
	inception_4a_1x1 = Activation('relu')(inception_4a_1x1)
	
	inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)
	
	#inception4e
	inception_4e_3x3 = Conv2D(160, (1,1), strides=(1,1), name='inception_4e_3x3_conv'+'1')(inception_4a)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'1')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
	inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
	inception_4e_3x3 = Conv2D(256, (3,3), strides=(2,2), name='inception_4e_3x3_conv'+'2')(inception_4e_3x3)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'2')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
	
	inception_4e_5x5 = Conv2D(64, (1,1), strides=(1,1), name='inception_4e_5x5_conv'+'1')(inception_4a)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'1')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
	inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
	inception_4e_5x5 = Conv2D(128, (5,5), strides=(2,2), name='inception_4e_5x5_conv'+'2')(inception_4e_5x5)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'2')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
	
	inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
	inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)
	
	inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)
	
	#inception5a
	inception_5a_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_3x3_conv'+'1')(inception_4e)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'1')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)
	inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
	inception_5a_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5a_3x3_conv'+'2')(inception_5a_3x3)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'2')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)
	
	inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
	inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
	
	inception_5a_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_pool_conv'+'')(inception_5a_pool)
	inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_pool_bn'+'')(inception_5a_pool)
	inception_5a_pool = Activation('relu')(inception_5a_pool)
	inception_5a_pool = ZeroPadding2D(padding=(1,1))(inception_5a_pool)
	
	inception_5a_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5a_1x1_conv'+'')(inception_4e)
	inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_1x1_bn'+'')(inception_5a_1x1)
	inception_5a_1x1 = Activation('relu')(inception_5a_1x1)
	
	inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)
	
	#inception_5b
	inception_5b_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_3x3_conv'+'1')(inception_5a)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'1')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)
	inception_5b_3x3 = ZeroPadding2D(padding=(1,1))(inception_5b_3x3)
	inception_5b_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5b_3x3_conv'+'2')(inception_5b_3x3)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'2')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)
	
	inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
	
	inception_5b_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_pool_conv'+'')(inception_5b_pool)
	inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_pool_bn'+'')(inception_5b_pool)
	inception_5b_pool = Activation('relu')(inception_5b_pool)
	
	inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)
	
	inception_5b_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5b_1x1_conv'+'')(inception_5a)
	inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_1x1_bn'+'')(inception_5b_1x1)
	inception_5b_1x1 = Activation('relu')(inception_5b_1x1)
	
	inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)
	
	av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
	reshape_layer = Flatten()(av_pool)
	dense_layer = Dense(128, name='dense_layer')(reshape_layer)
	norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)
	
	# Final Model
	model = Model(inputs=[myInput], outputs=norm_layer)
	return model

model = builtModel()
print("model built")

#------------------------

#https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view
model.load_weights('weights/openface_weights.h5')
print("weights loaded")

#------------------------
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output	
    
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    #euclidean_distance = l2_normalize(euclidean_distance)
    return euclidean_distance

#------------------------

metric = "cosine"  #cosine, euclidean

if metric == "cosine":
	threshold = 0.45
else:
	threshold = 0.95

#------------------------	

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "database/"

employees = dict()

for file in listdir(employee_pictures):
	employee, extension = file.split(".")
	img = preprocess_image('database/%s.jpg' % (employee))
	representation = model.predict(img)[0,:]
	
	employees[employee] = representation
	
print("employee representations retrieved successfully")

#------------------------

cap = cv2.VideoCapture(0) #webcam

while(True):
	ret, img = cap.read()
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: #discard small detected faces
			cv2.rectangle(img, (x,y), (x+w,y+h), color, 1) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (96, 96)) #resize to 96x96
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			#employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
			img_pixels /= 127.5
			img_pixels -= 1
			
			captured_representation = model.predict(img_pixels)[0,:]
			
			distances = []
			
			for i in employees:
				employee_name = i
				source_representation = employees[i]
				
				if metric == "cosine":
					distance = findCosineDistance(captured_representation, source_representation)
				elif metric == "euclidean":
					distance = findEuclideanDistance(captured_representation, source_representation)
				
				if dump:
					print(employee_name,": ",distance)
				distances.append(distance)
			
			label_name = 'unknown'
			index = 0
			for i in employees:
				employee_name = i
				if index == np.argmin(distances):
					if distances[index] <= threshold:
						#print("detected: ",employee_name)
						
						if metric == "euclidean":
							similarity = 100 + (90 - 100*distance)
						elif metric == "cosine":
							similarity = 100 + (40 - 100*distance)
						
						if similarity > 99.99: similarity = 99.99
						
						label_name = "%s (%s%s)" % (employee_name, str(round(similarity,2)), '%')
						
						break
					
				index = index + 1
			
			cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			
			if dump:
				print("----------------------")
					
			#connect face and text
			cv2.line(img,(x+w, y-64),(x+w-25, y-64),color,1)
			cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),color,1)
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()