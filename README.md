# TensorFlow 101: Introduction to Deep Learning

I have worked all my life in Machine Learning, and **I've never seen one algorithm knock over its benchmarks like Deep Learning** - Andrew Ng

This repository includes deep learning based project implementations I've done from scratch. You can find both the source code and documentation as a step by step tutorial. Model structrues and pre-trained weights are shared as well.

*PS: This repository is updated regularly. You should pull the repo if you forked.*

1- **Facial Expression Recognition** [`Code`](python/facial-expression-recognition.py), [`Tutorial`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/)

This is a custom CNN model. Kaggle [FER 2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) data set is fed to the model. This model runs fast and produces satisfactory results.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2017/12/pablo-facial-expression.png" width="70%" height="70%"></p>

2- **Real Time Facial Expression Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/emotion-analysis-from-video.py), [`Tutorial`](https://sefiks.com/2018/01/10/real-time-facial-expression-recognition-on-streaming-data/), [`Video 1`](https://youtu.be/Y7DfLvLKScs), [`Video 2`](https://youtu.be/Dm5ptTiIpkk)

This is the adaptation of the same model to real time video or web cam. [OpenCV](https://github.com/opencv/opencv)'s haar cascade module detects the face and we applied facial expression recoginition to detected face.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/01/real-time-emotion-mark.png" width="70%" height="70%"></p>

3- **Face Recognition With Oxford VGG-Face Model** [`Code`](python/vgg-face.ipynb), [`Tutorial`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)

Oxford Visual Geometry Group's VGG model is famous for confident scores on Imagenet contest. They [retrain](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) almost the same network structure for face recognition. Pre-trained weights are converted from Matlab to Keras. This implementation is mainly based on convolutional neural networks, autoencoders and transfer learning. This is on the top of the leaderboard for face recognition challenges.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/01/face-recognition-demo.png" width="70%" height="70%"></p>

4- **Real Time Deep Face Recognition Implementation with VGG-Face** [`Code`](python/deep-face-real-time.py), [`Video`](https://www.youtube.com/watch?v=tSU_lNi0gQQ)

This is the real time implementation of VGG-Face model for face recognition.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/04/real-time-face-recognition-demo.jpg" width="70%" height="70%"></p>

5- **Face Recognition with Google FaceNet Model** [`Code`](python/facenet.ipynb), [`Tutorial`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/facenet-real-time.py), [`Video`](https://youtu.be/vB1I5vWgTQg)

This is an alternative to Oxford VGG model. My experiments show that it runs faster than VGG-Face but it is less accurate even though FaceNet has a more complex structure. Pre-trained weights are converted from TensorFlow in the study of [David Sandberg](https://github.com/davidsandberg/facenet) to Keras. Besides, model construction is mainly based on the study of [Yu-Yang Huang](https://github.com/yuyang-huang/keras-inception-resnet-v2).

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/facenet-real-time-demo.jpg" width="70%" height="70%"></p>

6- **Face Recognition with OpenFace Model** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface.ipynb), [`Tutorial`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface-real-time.py), [`Video`](https://youtu.be/-4z2sL6wzP8)

[OpenFace](http://cmusatyalab.github.io/openface/) is a lightweight model for face recognition tasks. It is not the best but it is the fastest. Original study shared the pre-trained weights in Lua Torch format. On the other hand, [this repository](https://github.com/iwantooxxoox/Keras-OpenFace) is shared the pre-trained weights in a flat file. I build the OpenFace model in Keras based on these sources.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/openface-demo-cover.jpg" width="70%" height="70%"></p>

7- **Apparent Age and Gender Prediction** [`Code for age`](https://github.com/serengil/tensorflow-101/blob/master/python/apparent_age_prediction.ipynb), [`Code for gender`](https://github.com/serengil/tensorflow-101/blob/master/python/gender_prediction.ipynb), [`Tutorial`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/)

We've used VGG-Face model for apparent age prediction this time. We actually applied transfer learning. Locking the early layers' weights enables to have outcomes fast. 

This study is mainly based on [this study](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf). However, this study shared pre-trained weights in Caffe and use VGG model. I adapt VGG-Face instead of VGG and train the model from scratch. 

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/10/age-prediction-for-godfather-original.png" width="70%" height="70%"></p>

8- **Real Time Age and Gender Prediction** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py), [`Video`](https://youtu.be/tFI7vZn3P7E)

This is a real time apparent age and gender prediction implementation

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/02/age-real-time.jpg" width="50%" height="50%"></p>

9- **Celebrity You Look-Alike Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Find-Look-Alike-Celebrities.ipynb), [`Tutorial`](https://sefiks.com/2019/05/05/celebrity-look-alike-face-recognition-with-deep-learning-in-keras/)

Applying VGG-Face recognition technology for [imdb data set](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) will find your celebrity look-alike if you discard the threshold in similarity score.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/05/sefik-looks-alike-colin-hanks.jpg" width="50%" height="50%"></p>

10- **Real Time Celebrity Look-Alike Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/celebrity-look-alike-real-time.py), [`Video`](https://youtu.be/RMgIKU1H8DY)

This is the real time implementation of finding similar celebrities.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/05/celebrity-look-alike-real-time.jpg" width="70%" height="70%"></p>

11- **Race and Ethnicity Prediction** 
[`Tutorial`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/), [`AI Ethics Tutorial`](https://sefiks.com/2019/11/10/race-and-ethnicity-prediction-in-the-perspective-of-ai-ethics/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Race-Ethnicity-Prediction-Batch.ipynb), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/real-time-ethnicity-prediction.py), [`Video`](https://youtu.be/-ztiy5eJha8)

Ethnicity is a facial attribute as well and we can predict it from facial photos similiar to Age and Gender. We customize VGG-Face and applied transfer learning to classify 6 different race and ethnicity groups. [FairFace](https://github.com/joojs/fairface) data set is fed to the network to train.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2019/11/FairFace-testset.png" width="70%" height="70%"></p>

12- **Beauty Score Prediction** [`Tutorial`](https://sefiks.com/2019/12/25/beauty-score-prediction-with-deep-learning/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Attractive.ipynb)

South China University of Technology published a research paper about facial beauty prediction. They also [open-sourced](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) the data set. 60 labelers scored the beauty of 5500 people. We will build a regressor to find facial beauty score. We will also test the built regressor on a huge imdb data set to find the most beautiful ones.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/12/beauty-imdb.png" width="70%" height="70%"></p>

13- **Making Arts with Deep Learning: Artistic Style Transfer** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/style-transfer.ipynb), [`Tutorial`](https://sefiks.com/2018/07/20/artistic-style-transfer-with-deep-learning/), [`Video`](https://youtu.be/QKCcJVJ0DZA)

What if Vincent van Gogh had painted Istanbul Bosporus? Today we can answer this question. A deep learning technique named [artistic style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) enables to transform ordinary images to masterpieces.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2019/01/gsu_vincent.png" width="70%" height="70%"></p>

14- **Autoencoder and clustering** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Autoencoder.ipynb), [`Tutorial`](https://sefiks.com/2018/03/21/autoencoder-neural-networks-for-unsupervised-learning/)

We can use neural networks to represent data. If you design a neural networks model symmetric about the centroid and you can restore a base data with an acceptable loss, then output of the centroid layer can represent the base data. Representations can contribute any field of deep learning such as face recognition, style transfer or just clustering.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2018/03/autoencoder-and-autodecoder.png" width="70%" height="70%"></p>

15- **Convolutional Autoencoder and clustering** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/ConvolutionalAutoencoder.ipynb), [`Tutorial`](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/)

We can adapt same representation approach to convolutional neural networks, too.

16- **Transfer Learning: Consuming InceptionV3 to Classify Cat and Dog Images in Keras** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/transfer_learning.py), [`Tutorial`](https://sefiks.com/2017/12/10/transfer-learning-in-keras-using-inception-v3/)

We can have the outcomes of the other researchers effortlessly. Google researchers compete on Kaggle Imagenet competition. They got 97% accuracy. We will adapt Google's Inception V3 model to classify objects.

<p align="center"><img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2017/12/inceptionv3-result.png" width="70%" height="70%"></p>

17- **Handwritten Digit Classification Using Neural Networks** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/HandwrittenDigitsClassification.py), [`Tutorial`](https://sefiks.com/2017/09/11/handwritten-digit-classification-with-tensorflow/)

We had to apply feature extraction on data sets to use neural networks. Deep learning enables to skip this step. We just feed the data, and deep neural networks can extract features on the data set. Here, we will feed handwritten digit data (MNIST) to deep neural networks, and expect to learn digits.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2017/09/mnist-sample-output.png" width="70%" height="70%"></p>

18- **Handwritten Digit Recognition Using Convolutional Neural Networks with Keras** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/HandwrittenDigitRecognitionUsingCNNWithKeras.py), [`Tutorial`](https://sefiks.com/2017/11/05/handwritten-digit-recognition-using-cnn-with-keras/)

Convolutional neural networks are close to human brain. People look for some patterns in classifying objects. For example, mouth, nose and ear shape of a cat is enough to classify a cat. We don't look at all pixels, just focus on some area. Herein, CNN applies some filters to detect these kind of shapes. They perform better than conventional neural networks. Herein, we got almost 2% accuracy than fully connected neural networks.

19- **Automated Machine Learning and Auto-Keras for Image Data** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Auto-Keras.ipynb), [`Model`](https://github.com/serengil/tensorflow-101/blob/master/model/fer_keras_model_from_autokeras.json), [`Tutorial`](https://sefiks.com/2019/04/08/a-gentle-introduction-to-auto-keras/)

AutoML concept aims to find the best network structure and hyper-parameters. Here, I've applied AutoML to facial expression recognition data set. My custom design got 57% accuracy whereas AutoML found a better model and got 66% accuracy. This means almost 10% improvement in the accuracy.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/04/google-automl.jpg" width="70%" height="70%"></p>

20- **Explaining Deep Learning Models with SHAP** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/SHAP-Explainable-AI.ipynb), [`Tutorial`](https://sefiks.com/2019/07/01/how-shap-can-keep-you-from-black-box-ai/)

SHAP explains black box machine learning models and makes them transparent, explainable and provable.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/fer-for-shap.png" width="70%" height="70%"></p>

21- **Gradient Vanishing Problem** [`Code`](python/gradient-vanishing.py) [`Tutorial`](https://sefiks.com/2018/05/31/an-overview-to-gradient-vanishing-problem/)

Why legacy activation functions such as sigmoid and tanh disappear on the pages of the history?

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/07/gradient-vanishing-problem-summary.jpg" width="70%" height="70%"></p>

22- **How single layer perceptron works** [`Code`](python/single-layer-perceptron.py)

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

There are many ways to support a project - starring the GitHub repos is one.

# Licence

This repository is licensed under MIT license - see [`LICENSE`](https://github.com/serengil/tensorflow-101/blob/master/LICENSE) for more details
