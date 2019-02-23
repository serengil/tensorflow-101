import numpy as np
import cv2
from keras.preprocessing import image
import time

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('C:/Users/IS96273/AppData/Local/Continuum/anaconda3/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#cap = cv2.VideoCapture('zuckerberg.mp4') #process videos
cap = cv2.VideoCapture(0) #process real time web-cam

frame = 0

while(True):
	ret, img = cap.read()
	
	img = cv2.resize(img, (640, 360))
	img = img[0:308,:]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		if w > 130: #trick: ignore small faces
			#cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			#------------------------------
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			max_index = np.argmax(predictions[0])
			
			#background of expression list
			overlay = img.copy()
			opacity = 0.4
			cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
			cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
			
			#connect face and expressions
			cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
			cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
			
			emotion = ""
			for i in range(len(predictions[0])):
				emotion = "%s %s%s" % (emotions[i], round(predictions[0][i]*100, 2), '%')
				
				"""if i != max_index:
					color = (255,0,0)"""
					
				color = (255,255,255)
				
				cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
				
			#-------------------------
	
	cv2.imshow('img',img)
	
	frame = frame + 1
	#print(frame)
	
	#---------------------------------
	
	if frame > 227:
		break
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things
cap.release()
cv2.destroyAllWindows()
