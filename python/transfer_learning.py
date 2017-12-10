from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
#--------------------------------

store_model = False

#--------------------------------

if store_model == True:
	model = InceptionV3(weights='imagenet', include_top=True)
	
	#save model and weights
	model_config = model.to_json()
	open("inceptionv3_structure.json", "w").write(model_config)
	model.save_weights('inceptionv3_weights.h5')
else:
	from keras.models import model_from_json
	model = model_from_json(open("inceptionv3_structure.json", "r").read())
	model.load_weights('inceptionv3_weights.h5')
	print("inception v3 model loaded")
	
#print("model structure: ", model.summary())
#print("model weights: ", model.get_weights())

#put images in testset folder, name images from 1.jpg to 16.jpg
for i in range(1, 17):
	
	img_path = 'testset/%s.jpg' % (i)
	
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	
	features = model.predict(x)
	print(decode_predictions(features, top = 3))
	
	plt.imshow(image.load_img(img_path))
	plt.show()
