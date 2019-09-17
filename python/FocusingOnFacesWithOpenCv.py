import cv2
import time
import numpy as np

#-----------------------

time_threshold = 10
frame_threshold = 15
margin = 10

#-----------------------

freeze = False
face_detected = False; face_included_frames = 0
tic = time.time()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) #webcam

while(True):
	ret, img = cap.read()
	raw_img = img.copy()
	resolution = img.shape #(480, 640, 3)

	if freeze == False: 
		faces = face_cascade.detectMultiScale(img, 1.3, 5)
		
		if len(faces) == 0:
			face_included_frames = 0
	else: 
		faces = []
	
	detected_faces = []
	face_index = 0
	for (x,y,w,h) in faces:
		if w > 130: #discard small detected faces
			
			face_detected = True
			if face_index == 0:
				face_included_frames = face_included_frames + 1 #increase frame for a single face
			
			cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
			
			cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			
			#add margin to detected face
			try:
				margin_x = int((w * margin)/100)
				margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+2*margin_y), int(x-margin_x):int(x+w+2*margin_x)]
				
				detected_faces.append((x-margin_x,y-margin_y,w+2*margin_x,h+2*margin_y))
				
			except:
				detected_faces.append((x,y,w,h))
			face_index = face_index + 1
			#detected_faces.append((x,y,w,h))
			
	if face_detected == True and face_included_frames == frame_threshold and freeze == False:
		freeze = True
		#base_img = img.copy()
		base_img = raw_img.copy()
		detected_faces_final = detected_faces.copy()
		tic = time.time()
	
	if freeze == True:
		
		nn_tic = time.time()
		#--------------------------------------
		#put face recognition code block here
		
		#--------------------------------------
		nn_toc = time.time()
		#nn_toc - nn_tic is time for face recognition, ignore it
		
		toc = time.time()
		if (toc - tic) - (nn_toc - nn_tic) < time_threshold:
			freeze_img = base_img.copy()
			#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue
			
			for detected_face in detected_faces_final:
				x_offset = detected_face[0]
				y_offset = detected_face[1]
				w_offset = detected_face[2]
				h_offset = detected_face[3]
				
				cv2.rectangle(freeze_img, (x_offset,y_offset), (x_offset+w_offset,y_offset+h_offset), (67,67,67), 2) #draw rectangle to main image
			
				custom_face = base_img[y_offset:y_offset+h_offset, x_offset:x_offset+w_offset]
				freeze_img[y_offset:y_offset+h_offset, x_offset:x_offset+w_offset] = custom_face
			
			time_left = int(time_threshold - (toc - tic) + 1)
			#print(time_left)
			cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.imshow('img', freeze_img)
		else:
			face_detected = False
			face_included_frames = 0
			freeze = False
		
	else:
		cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()