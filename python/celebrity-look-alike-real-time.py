import os

import pandas as pd
import numpy as np
import scipy.io

import time

from PIL import Image

import cv2
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
#-----------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	#you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
	from keras.models import model_from_json
	model.load_weights('vgg_face_weights.h5')
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor

model = loadVggFaceModel()
print("vgg face model loaded")

#------------------------
exists = os.path.isfile('representations.pkl')

if exists != True: #initializations lasts almost 1 hour. but it can be run once.
	
	#download imdb data set here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ . Faces only version (7 GB)
	mat = scipy.io.loadmat('imdb_data_set/imdb.mat')
	print("imdb.mat meta data file loaded")

	columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]

	instances = mat['imdb'][0][0][0].shape[1]

	df = pd.DataFrame(index = range(0,instances), columns = columns)

	for i in mat:
		if i == "imdb":
			current_array = mat[i][0][0]
			for j in range(len(current_array)):
				#print(j,". ",columns[j],": ",current_array[j][0])
				df[columns[j]] = pd.DataFrame(current_array[j][0])

	print("data frame loaded (",df.shape,")")

	#-------------------------------

	#remove pictures does not include any face
	df = df[df['face_score'] != -np.inf]

	#some pictures include more than one face, remove them
	df = df[df['second_face_score'].isna()]

	#discard inclear ones
	df = df[df['face_score'] >= 5]

	#-------------------------------
	#some speed up tricks. this is not a must.

	#discard old photos
	df = df[df['photo_taken'] >= 2000]

	print("some instances ignored (",df.shape,")")
	#-------------------------------

	def extractNames(name):
		return name[0]

	df['celebrity_name'] = df['name'].apply(extractNames)

	def getImagePixels(image_path):
		return cv2.imread("imdb_data_set/%s" % image_path[0]) #pixel values in scale of 0-255

	tic = time.time()
	df['pixels'] = df['full_path'].apply(getImagePixels)
	toc = time.time()

	print("reading pixels completed in ",toc-tic," seconds...") #3.4 seconds

	def findFaceRepresentation(img):
		detected_face = img
		
		try: 
			detected_face = cv2.resize(detected_face, (224, 224))
			#plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
			
			#normalize detected face in scale of -1, +1

			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 127.5
			img_pixels -= 1
			
			representation = model.predict(img_pixels)[0,:]
		except:
			representation = None
			
		return representation

	tic = time.time()
	df['face_vector_raw'] = df['pixels'].apply(findFaceRepresentation) #vector for raw image
	toc = time.time()
	print("extracting face vectors completed in ",toc-tic," seconds...")

	tic = time.time()
	df.to_pickle("representations.pkl")
	toc = time.time()
	print("storing representations completed in ",toc-tic," seconds...")

else:
	#if you run to_pickle command once, then read pickle completed in seconds in your following runs
	tic = time.time()
	df = pd.read_pickle("representations.pkl")
	toc = time.time()
	print("reading pre-processed data frame completed in ",toc-tic," seconds...")

#-----------------------------------------

print("data set: ",df.shape)

cap = cv2.VideoCapture(0) #webcam

while(True):
	ret, img = cap.read()
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	resolution_x = img.shape[1]; resolution_y = img.shape[0]
	
	for (x,y,w,h) in faces:
		if w > 0:
			#cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1)
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			
			#add 5% margin around the face
			try:
				margin = 0 #5
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				if y-margin_y > 0 and x-margin_x > 0 and y+h+margin_y < resolution_y and x+w+margin_x < resolution_x:
					detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			#normalize in scale of [-1, +1]
			img_pixels /= 127.5
			img_pixels -= 1
			
			captured_representation = model.predict(img_pixels)[0,:]			
			#----------------------------------------------
			
			def findCosineSimilarity(source_representation, test_representation=captured_representation):
				try:
					a = np.matmul(np.transpose(source_representation), test_representation)
					b = np.sum(np.multiply(source_representation, source_representation))
					c = np.sum(np.multiply(test_representation, test_representation))
					return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
				except:
					return 10 #assign a large value. similar faces will have small value.
			
			df['similarity'] = df['face_vector_raw'].apply(findCosineSimilarity)
			
			#look-alike celebrity
			min_index = df[['similarity']].idxmin()[0]
			instance = df.ix[min_index]
			
			name = instance['celebrity_name']
			similarity = instance['similarity']
			similarity = (1 - similarity)*100
			
			#print(name," (",similarity,"%)")
			
			if similarity > 50:
				full_path = instance['full_path'][0]
				celebrity_img = cv2.imread("imdb_data_set/%s" % full_path)
				celebrity_img = cv2.resize(celebrity_img, (112, 112))
				
				try:	
					img[y-120:y-120+112, x+w:x+w+112] = celebrity_img

					label = name+" ("+"{0:.2f}".format(similarity)+"%)"
					cv2.putText(img, label, (x+w-10, y - 120 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
					
					#connect face and text
					cv2.line(img,(x+w, y-64),(x+w-25, y-64),(67,67,67),1)
					cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),(67,67,67),1)
				except Exception as e:
					print("exception occured: ", str(e))
	
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()