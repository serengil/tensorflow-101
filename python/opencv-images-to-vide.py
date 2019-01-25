import cv2
import numpy as np
 
frame_width = 913
frame_height = 684
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

s_img = cv2.imread("style-small.jpg")
x_offset=600; y_offset=25

terminate = False
#while(True):
for i in range(999, 1500):
  
  #putting generated_*.png files in a directory    
  file = "C:/python/ortakoy/generated_%d.png" % (i)
  print(file)
  frame = cv2.imread(file)
  #cv2.imshow('frame',frame)
  
  frame = cv2.resize(frame, (913, 684))

  fps = 5
    
  epoch = i-999
  label = 'Frame %d' % (epoch+1)
  
  for j in range(fps):
    
    cv2.putText(frame, label, (int(50), int(630)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Write the frame into the file 'output.avi'
    out.write(frame)
  
    # Display the resulting frame    
    cv2.imshow('frame',frame)
  
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      terminate = True
      break
  
  if terminate == True:
    break
    
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 