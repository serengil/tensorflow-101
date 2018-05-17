#This program applies convolutional neural networks to handwritten digit dataset (MNIST)
#It consumes Keras API, uses TensorFlow as backend
#It produces 99.29% accuracy on test set

#blog post: https://sefiks.com/2017/11/05/handwritten-digit-recognition-using-cnn-with-keras/

#---------------------------------------
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

num_classes = 10 #results can be from 0 to 9

#you might change batch size and epoch to monitor the effect on system.
#more successful results can be produced if optimum batch size and epoch values found
batch_size = 250
epochs = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#---------------------------------------

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #transform 2D 28x28 matrix to 3D (28x28x1) matrix
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255 #inputs have to be between [0, 1]
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#---------------------------------------

# convert labels to binary form
y_train = keras.utils.to_categorical(y_train, num_classes) #e.g. label 2 would be represented as 0010000000
y_test = keras.utils.to_categorical(y_test, num_classes)

#---------------------------------------
#create neural networks structure
model = Sequential()

#1st convolution layer
model.add(Conv2D(32, (3, 3) #32 is number of filters and (3, 3) is the size of the filter.
	, input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3))) # apply 64 filters sized of (3x3) on 2nd convolution layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer. 1 hidden layer consisting of 512 nodes
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

#---------------------------------------

#Train the model with small size instances. Thus, you can create a model with a single CPU in a short time.
gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

#---------------------------------------

model.compile(loss='categorical_crossentropy'
	, optimizer=keras.optimizers.Adam()
	, metrics=['accuracy']
)

model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs, 
	validation_data=(x_test, y_test) #validate on all test set
)
model.save("model.hdf5")
#---------------------------------------

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

#---------------------------------------
model = load_model("model.hdf5")

predictions = model.predict(x_test)

#display wrongly classified instances
index = 0
for i in predictions:
	if index < 10000:
		actual = np.argmax(y_test[index])
		pred = np.argmax(i)
		
		if actual != pred:
			print("predict: ",pred," actual: ",actual)
			picture = x_test[index]
			picture = picture.reshape([28, 28]);
			plt.gray()
			plt.imshow(picture)
			plt.show()
		
	index = index + 1
