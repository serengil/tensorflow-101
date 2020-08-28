import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

#--------------------------------------------

def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
	eye_detector_path = path+"/data/haarcascade_eye.xml"
	
	if os.path.isfile(face_detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
	
	return path+"/data/"

#--------------------------------------------

ssd_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

opencv_path = get_opencv_path()
haar_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
haar_detector = cv2.CascadeClassifier(haar_detector_path)

#--------------------------------------------

ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

#--------------------------------------------

cap = cv2.VideoCapture('zuckerberg.mp4')

#detector_model = 'ssd'
detector_model = 'haar'

quit = False
tic = time.time()
frame = 0
while(True):
	
	if frame % 100 == 0:
		toc = time.time()
		print(frame,", ",toc-tic)
		tic = time.time()
	
	ret, img = cap.read()
	
	try:
		original_size = img.shape
		
		cv2.putText(img, detector_model, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
	
		if detector_model == 'ssd':
			target_size = (300, 300)
			
			base_img = img.copy() #high resolution image
			
			img = cv2.resize(img, target_size)
			
			aspect_ratio_x = (original_size[1] / target_size[1])
			aspect_ratio_y = (original_size[0] / target_size[0])
			
			imageBlob = cv2.dnn.blobFromImage(image = img)

			ssd_detector.setInput(imageBlob)
			detections = ssd_detector.forward()
			
			detections_df = pd.DataFrame(detections[0][0], columns = ssd_labels)
			
			detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
			detections_df = detections_df[detections_df['confidence'] >= 0.90]
			
			detections_df['left'] = (detections_df['left'] * 300).astype(int)
			detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
			detections_df['right'] = (detections_df['right'] * 300).astype(int)
			detections_df['top'] = (detections_df['top'] * 300).astype(int)
			
			for i, instance in detections_df.iterrows():
				confidence_score = str(round(100*instance["confidence"], 2))+" %"
				left = instance["left"]
				right = instance["right"]
				bottom = instance["bottom"]
				top = instance["top"]
				
				detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
				
				if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
					
					cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
					
					cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image
					
					img = base_img.copy()
				
		elif detector_model == 'haar':
			
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			faces = haar_detector.detectMultiScale(gray, 1.3, 5)
			
			for (x,y,w,h) in faces:
				if w > 0:
					cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,255), 1) #highlight detected face
			
		#----------------------------
	
		#cv2.imshow('img',img)
		cv2.imwrite( "%s/%s.jpg" % (detector_model, str(frame)), img );
		frame = frame + 1
	except:
		quit = True
	
	if quit == True or (cv2.waitKey(1) & 0xFF == ord('q')): #press q to quit
		toc = time.time()
		
		print(frame," frames process in ",toc-tic," seconds")
		break

#kill open cv things
cap.release()
cv2.destroyAllWindows()