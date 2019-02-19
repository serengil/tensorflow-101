#Documentation: https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/

import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir

#-----------------------
#you can find male and female icons here: https://github.com/serengil/tensorflow-101/tree/master/dataset

enableGenderIcons = True

male_icon = cv2.imread("male.jpg")
male_icon = cv2.resize(male_icon, (40, 40))

female_icon = cv2.imread("female.jpg")
female_icon = cv2.resize(female_icon, (40, 40))
#-----------------------

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
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
	
	return model

def ageModel():
	model = loadVggFaceModel()
	
	base_model_output = Sequential()
	base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)
	
	age_model = Model(inputs=model.input, outputs=base_model_output)
	
	#you can find the pre-trained weights for age prediction here: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
	age_model.load_weights("age_model_weights.h5")
	
	return age_model

def genderModel():
	model = loadVggFaceModel()
	
	base_model_output = Sequential()
	base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	gender_model = Model(inputs=model.input, outputs=base_model_output)
	
	#you can find the pre-trained weights for gender prediction here: https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
	gender_model.load_weights("gender_model_weights.h5")
	
	return gender_model
	
age_model = ageModel()
gender_model = genderModel()

#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

#------------------------

cap = cv2.VideoCapture(0) #capture webcam

while(True):
	ret, img = cap.read()
	#img = cv2.resize(img, (640, 360))
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: #ignore small faces
			
			#mention detected face
			"""overlay = img.copy(); output = img.copy(); opacity = 0.6
			cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),cv2.FILLED) #draw rectangle to main image
			cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)"""
			cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1) #draw rectangle to main image
			
			#extract detected face
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			
			try:
				#age gender data set has 40% margin around the face. expand detected face.
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			try:
				#vgg-face expects inputs (224, 224, 3)
				detected_face = cv2.resize(detected_face, (224, 224))
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				#find out age and gender
				age_distributions = age_model.predict(img_pixels)
				apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
				
				gender_distribution = gender_model.predict(img_pixels)[0]
				gender_index = np.argmax(gender_distribution)
				
				if gender_index == 0: gender = "F"
				else: gender = "M"
			
				#background for age gender declaration
				info_box_color = (46,200,255)
				#triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
				
				#labels for age and gender
				cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
				if enableGenderIcons:
					if gender == 'M': gender_icon = male_icon
					else: gender_icon = female_icon
					
					img[y-75:y-75+male_icon.shape[0], x+int(w/2)-45:x+int(w/2)-45+male_icon.shape[1]] = gender_icon
				else:
					cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
			except Exception as e:
				print("exception",str(e))
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()
