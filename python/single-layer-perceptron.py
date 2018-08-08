import numpy as np

operator = 'and'

#----------------

atributes = np.array([ [0, 0], [0, 1], [1, 0], [1, 1]])

if operator == 'and':
	labels = np.array([0, 0, 0, 1])
elif operator == 'or':
	labels = np.array([0, 1, 1, 1])
elif operator == 'xor':
	labels = np.array([0, 1, 1, 0])

#----------------

w = [+9, +9] #initial random values for weights

threshold = 5
alpha = 0.5 #learning rate
epoch = 1000 #learning time
#----------------

print("learning rate: ", alpha,", threshold: ", threshold)

for i in range(0, epoch):
	print("epoch ", i+1)
	global_delta = 0 #this variable is used to terminate the for loop if learning completed in early epoch
	for j in range(len(atributes)):
		
		actual = labels[j]
		
		sum = atributes[j][0]*w[0] + atributes[j][1]*w[1]
		
		if sum > threshold: #then fire
			predicted = 1
		else: #do not fire
			predicted = 0
		
		delta = actual - predicted
		global_delta = global_delta + abs(delta)
		
		#update weights with respect to the error
		for k in range(0, 2):
			w[k] = w[k] + delta * alpha
				
		print(atributes[j][0]," ", operator, " ", atributes[j][1], " -> actual: ", actual, ", predicted: ", predicted, " (w: ",w[0],")")
		
	if global_delta == 0:
		break
	
	print("------------------------------")
