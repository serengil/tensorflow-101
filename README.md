# TensorFlow 101: Introduction to Deep Learning

I have worked all my life in Machine Learning, and **I've never seen one algorithm knock over its benchmarks like Deep Learning** - Andrew Ng

This repository includes deep learning based project implementations I've done from scratch. You can find both the source code and documentation as a step by step tutorial. Model structrues and pre-trained weights are shared as well.

**Facial Expression Recognition** [`Code`](python/facial-expression-recognition.py), [`Tutorial`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/emotion-analysis-from-video.py), [`Video`](https://youtu.be/Y7DfLvLKScs)

This is a custom CNN model. Kaggle [FER 2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) data set is fed to the model. This model runs fast and produces satisfactory results. It can be also run real time as well.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2017/12/pablo-facial-expression.png" width="70%" height="70%"></p>

**Face Recognition** [`VGG-Face Code`](https://github.com/serengil/tensorflow-101/blob/master/python/vgg-face.ipynb), [`VGG-Face Tutorial`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Facenet Code`](https://github.com/serengil/tensorflow-101/blob/master/python/facenet.ipynb), [`Facenet Tutorial`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface.ipynb), [`OpenFace Tutorial`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/)

We've mentioned one of the most successful face recognition models. Oxford Visual Geometry Group (VGG) developed VGG-Face model. This model is also the winner of imagenet competition. They just tuned the weights of the same imagenet model to detect facial attributes. Moreover, Google announced its face recognition model Facenet. Furthermore, Carnegie Mellon University open-sourced its face recognition model OpenFace. 

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/01/face-recognition-demo.png" width="70%" height="70%"></p>

**Real Time Deep Face Recognition Implementation** 
[`VGG-Face Code`](https://github.com/serengil/tensorflow-101/blob/master/python/deep-face-real-time.py), [`VGG-Face Video`](https://www.youtube.com/watch?v=tSU_lNi0gQQ), [`Facenet Code`](https://github.com/serengil/tensorflow-101/blob/master/python/facenet-real-time.py), [`Facenet Video`](https://youtu.be/vB1I5vWgTQg), [`OpenFace Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface-real-time.py), [`OpenFace Video`](https://youtu.be/-4z2sL6wzP8)

These are the real time implementations of the common face recognition models we've mentioned in the previous section. VGG-Face has the highest face recognition score but it comes with the high complexity among models. On the other hand, OpenFace is a pretty model and it has a close accuracy to VGG-Face but its simplicity offers high speed than others.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/openface-demo-cover.jpg" width="70%" height="70%"></p>

**Apparent Age and Gender Prediction** [`Tutorial`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`Code for age`](https://github.com/serengil/tensorflow-101/blob/master/python/apparent_age_prediction.ipynb), [`Code for gender`](https://github.com/serengil/tensorflow-101/blob/master/python/gender_prediction.ipynb), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py), [`Video`](https://youtu.be/tFI7vZn3P7E)

We've used VGG-Face model for apparent age prediction this time. We actually applied transfer learning. Locking the early layers' weights enables to have outcomes fast. 

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/10/age-prediction-for-godfather-original.png" width="70%" height="70%"></p>

**Celebrity You Look-Alike Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Find-Look-Alike-Celebrities.ipynb), [`Tutorial`](https://sefiks.com/2019/05/05/celebrity-look-alike-face-recognition-with-deep-learning-in-keras/), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/celebrity-look-alike-real-time.py), [`Video`](https://youtu.be/RMgIKU1H8DY)

Applying VGG-Face recognition technology for [imdb data set](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) will find your celebrity look-alike if you discard the threshold in similarity score.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/05/sefik-looks-alike-colin-hanks.jpg" width="50%" height="50%"></p>

**Race and Ethnicity Prediction** 
[`Tutorial`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Race-Ethnicity-Prediction-Batch.ipynb), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/real-time-ethnicity-prediction.py), [`Video`](https://youtu.be/-ztiy5eJha8)

Ethnicity is a facial attribute as well and we can predict it from facial photos. We customize VGG-Face and we also applied transfer learning to classify 6 different ethnicity groups.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2019/11/FairFace-testset.png" width="70%" height="70%"></p>

**Beauty Score Prediction** [`Tutorial`](https://sefiks.com/2019/12/25/beauty-score-prediction-with-deep-learning/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Attractive.ipynb)

South China University of Technology published a research paper about facial beauty prediction. They also [open-sourced](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) the data set. 60 labelers scored the beauty of 5500 people. We will build a regressor to find facial beauty score. We will also test the built regressor on a huge imdb data set to find the most beautiful ones.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2020/01/beauty-imdb-v2.png" width="70%" height="70%"></p>

**Making Arts with Deep Learning: Artistic Style Transfer** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/style-transfer.ipynb), [`Tutorial`](https://sefiks.com/2018/07/20/artistic-style-transfer-with-deep-learning/), [`Video`](https://youtu.be/QKCcJVJ0DZA)

What if Vincent van Gogh had painted Istanbul Bosporus? Today we can answer this question. A deep learning technique named [artistic style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) enables to transform ordinary images to masterpieces.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2019/01/gsu_vincent.png" width="70%" height="70%"></p>

**Autoencoder and clustering** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Autoencoder.ipynb), [`Tutorial`](https://sefiks.com/2018/03/21/autoencoder-neural-networks-for-unsupervised-learning/)

We can use neural networks to represent data. If you design a neural networks model symmetric about the centroid and you can restore a base data with an acceptable loss, then output of the centroid layer can represent the base data. Representations can contribute any field of deep learning such as face recognition, style transfer or just clustering.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2018/03/autoencoder-and-autodecoder.png" width="70%" height="70%"></p>

**Convolutional Autoencoder and clustering** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/ConvolutionalAutoencoder.ipynb), [`Tutorial`](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/)

We can adapt same representation approach to convolutional neural networks, too.

**Transfer Learning: Consuming InceptionV3 to Classify Cat and Dog Images in Keras** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/transfer_learning.py), [`Tutorial`](https://sefiks.com/2017/12/10/transfer-learning-in-keras-using-inception-v3/)

We can have the outcomes of the other researchers effortlessly. Google researchers compete on Kaggle Imagenet competition. They got 97% accuracy. We will adapt Google's Inception V3 model to classify objects.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2017/12/inceptionv3-result.png" width="70%" height="70%"></p>

**Handwritten Digit Classification Using Neural Networks** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/HandwrittenDigitsClassification.py), [`Tutorial`](https://sefiks.com/2017/09/11/handwritten-digit-classification-with-tensorflow/)

We had to apply feature extraction on data sets to use neural networks. Deep learning enables to skip this step. We just feed the data, and deep neural networks can extract features on the data set. Here, we will feed handwritten digit data (MNIST) to deep neural networks, and expect to learn digits.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2017/09/mnist-sample-output.png" width="70%" height="70%"></p>

**Handwritten Digit Recognition Using Convolutional Neural Networks with Keras** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/HandwrittenDigitRecognitionUsingCNNWithKeras.py), [`Tutorial`](https://sefiks.com/2017/11/05/handwritten-digit-recognition-using-cnn-with-keras/)

Convolutional neural networks are close to human brain. People look for some patterns in classifying objects. For example, mouth, nose and ear shape of a cat is enough to classify a cat. We don't look at all pixels, just focus on some area. Herein, CNN applies some filters to detect these kind of shapes. They perform better than conventional neural networks. Herein, we got almost 2% accuracy than fully connected neural networks.

**Automated Machine Learning and Auto-Keras for Image Data** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Auto-Keras.ipynb), [`Model`](https://github.com/serengil/tensorflow-101/blob/master/model/fer_keras_model_from_autokeras.json), [`Tutorial`](https://sefiks.com/2019/04/08/a-gentle-introduction-to-auto-keras/)

AutoML concept aims to find the best network structure and hyper-parameters. Here, I've applied AutoML to facial expression recognition data set. My custom design got 57% accuracy whereas AutoML found a better model and got 66% accuracy. This means almost 10% improvement in the accuracy.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/04/google-automl.jpg" width="70%" height="70%"></p>

**Explaining Deep Learning Models with SHAP** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/SHAP-Explainable-AI.ipynb), [`Tutorial`](https://sefiks.com/2019/07/01/how-shap-can-keep-you-from-black-box-ai/)

SHAP explains black box machine learning models and makes them transparent, explainable and provable.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/fer-for-shap.png" width="70%" height="70%"></p>

**Gradient Vanishing Problem** [`Code`](python/gradient-vanishing.py) [`Tutorial`](https://sefiks.com/2018/05/31/an-overview-to-gradient-vanishing-problem/)

Why legacy activation functions such as sigmoid and tanh disappear on the pages of the history?

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/gradient-vanishing-problem-summary.jpg" width="70%" height="70%"></p>

**How single layer perceptron works** [`Code`](python/single-layer-perceptron.py)

This is the 1957 model implementation of the perceptron.

<p align="center"><img src="https://i1.wp.com/sefiks.com/wp-content/uploads/2018/05/perceptron.png" width="50%" height="50%"></p>


# Requirements

I have tested this repository on the following environments. To avoid environmental issues, confirm your environment is same as below.

```
C:\>python --version
Python 3.6.4 :: Anaconda, Inc.

C:\>activate tensorflow

(tensorflow) C:\>python
Python 3.5.5 |Anaconda, Inc.| (default, Apr  7 2018, 04:52:34) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
1.9.0
>>>
>>> import keras
Using TensorFlow backend.
>>> print(keras.__version__)
2.2.0
>>>
>>> import cv2
>>> print(cv2.__version__)
3.4.4
```

To get your environment up from zero, you can follow the instructions in the following videos.

**Installing TensorFlow and Prerequisites** [`Video`](https://www.youtube.com/watch?v=JeR2M46tLlE)

**Installing Keras** [`Video`](https://www.youtube.com/watch?v=qx5pivWvKC8)

# Disclaimer

This repo might use some external sources. Notice that related tutorial links and comments in the code blocks cite references already.

# Support

There are many ways to support a project - starring⭐️ the GitHub repos is one.

# Licence

This repository is licensed under MIT license - see [`LICENSE`](https://github.com/serengil/tensorflow-101/blob/master/LICENSE) for more details
