import os
from os import listdir
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt

#-----------------------

case = 'background' #background, face, pixelated
backend = 'mtcnn' #haar, mtcnn
mode = 'display' #record, display

#-----------------------
def blur_img(img, factor = 20):

    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)
	
    #ensure the shape of the kernel is odd
    if kW % 2 == 0: kW -= 1
    if kH % 2 == 0: kH -= 1
    
    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
	
    return blurred_img

def extract_indexes(length, step_size):
    
    indexes = []
    
    cycles = int(length / step_size)

    for i in range(cycles):
        begin = i * step_size; end = i * step_size+step_size
        #print(i, ". [",begin,", ",end,")")
        
        index = []
        index.append(begin)
        index.append(end)
        indexes.append(index)

        if begin >= length: break
        if end > length: end = length

    if end < length:
        #print(i+1,". [", end,", ",length,")")
        index = []
        index.append(end)
        index.append(length)
        indexes.append(index)
    
    return indexes

detector = MTCNN()

#-----------------------

if backend == 'haar':
	#OpenCV haarcascade module

	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder

	detector_path = path+"/data/haarcascade_frontalface_default.xml"

	if os.path.isfile(detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")
	else:
		face_cascade = cv2.CascadeClassifier(detector_path)
#------------------------

cap = cv2.VideoCapture("zuckerberg.mp4") #0 for webcam or video
frame = 0
while(True):
	ret, img = cap.read()
	
	if backend == 'haar':
		faces = face_cascade.detectMultiScale(img, 1.3, 5)
	elif backend == 'mtcnn':
		faces = detector.detect_faces(img)
	
	base_img = img.copy()
	
	if case == 'background':
		img = blur_img(img, factor = 70)
	
	for detection in faces:
		
		if backend == 'mtcnn':
			score = detection["confidence"]
			x, y, w, h = detection["box"]
		elif backend == 'haar':
			x,y,w,h = detection
			
		if (backend == 'haar' and w > 0) or (backend == 'mtcnn' and w > 0 and score >= 0.90):
			
			detected_face = base_img[int(y):int(y+h), int(x):int(x+w)]
			
			if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
			
				if case == 'background':
					img[y:y+h, x:x+w] = detected_face
				elif case == 'face':
					detected_face_blurred = blur_img(detected_face, factor = 3)
					img[y:y+h, x:x+w] = detected_face_blurred
				elif case == 'pixelated':
					
					pixelated_face = detected_face.copy()
					
					step_size = 25
					
					width = pixelated_face.shape[0]
					height = pixelated_face.shape[1]
					
					#---------------------------------
					
					iteration = 0
					for wi in extract_indexes(width, step_size):
						for hi in extract_indexes(height, step_size):
							
							detected_face_area = detected_face[wi[0]:wi[1], hi[0]:hi[1]]
							#print(width,"x",height,": ",wi,", ",hi," (",detected_face_area.shape)
							
							factor = 0.5# if iteration % 1 == 0 else 1
							
							if detected_face_area.shape[0] > 0 and detected_face_area.shape[1] > 0:
								detected_face_area = blur_img(detected_face_area, factor = factor)
								pixelated_face[wi[0]:wi[1], hi[0]:hi[1]] = detected_face_area
							
							iteration = iteration + 1
				
					img[y:y+h, x:x+w] = pixelated_face
	
	if mode == 'display':
		cv2.imshow('img',img)
	elif mode == 'record':
		cv2.imwrite('outputs_%s/%d.png' % (backend, frame), img)
		if frame % 50 == 0: print(frame)
	
	frame = frame + 1
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()
