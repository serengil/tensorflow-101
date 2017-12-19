import numpy as np

operator = 'and'

atributes = [
	[0, 0]
	, [0, 1]
	, [1, 0]
	, [1, 1]
]

if operator == 'and':
	labels = [0, 0, 0, 1]
elif operator == 'or':
	labels = [0, 1, 1, 1]

w = [-1, -1, -1] #initial values for weights

#----------------
threshold = 5
alpha = 0.5 #learning rate
epoch = 1000
bias = 1
#----------------

print("learning rate: ", alpha,", threshold: ", threshold)

data = np.array(atributes, 'int64')
target = np.array(labels, 'int64')

for i in range(0, epoch):
	print("epoch ", i+1)
	index = 0
	global_delta = 0
	for j in range(len(atributes)):
		
		actual = labels[index]
		
		sum = data[index][0]*w[0] + data[index][1]*w[1] + bias * w[2]
		
		if sum > threshold:
			predicted = 1
		else:
			predicted = 0
		
		delta = actual - predicted
		global_delta = global_delta + abs(delta)
		
		for k in range(0, 3):
			w[k] = w[k] + delta * alpha
				
		print(data[index][0]," ", operator, " ", data[index][1], " -> actual: ", actual, ", predicted: ", predicted, " (w: ",w[0],")")
		
		index = index + 1
		
	if global_delta == 0:
		break
	
	print("------------------------------")
	
