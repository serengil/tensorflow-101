# TensorFlow 101: Introduction to Deep Learning

[![Stars](https://img.shields.io/github/stars/serengil/tensorflow-101)](https://github.com/serengil/tensorflow-101)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/tensorflow-101/blob/master/LICENSE)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dserengil%26type%3Dpatrons&style=flat)](https://www.patreon.com/serengil?repo=tensorflow101)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/serengil?logo=GitHub&color=lightgray)](https://github.com/sponsors/serengil)

I have worked all my life in Machine Learning, and **I've never seen one algorithm knock over its benchmarks like Deep Learning** - Andrew Ng

This repository includes deep learning based project implementations I've done from scratch. You can find both the source code and documentation as a step by step tutorial. Model structrues and pre-trained weights are shared as well.

**Facial Expression Recognition** [`Code`](python/facial-expression-recognition.py), [`Tutorial`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/)

This is a custom CNN model. Kaggle [FER 2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) data set is fed to the model. This model runs fast and produces satisfactory results. It can be also run real time as well.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2017/12/pablo-facial-expression.png" width="70%" height="70%"></p>

We can run emotion analysis in real time as well [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/emotion-analysis-from-video.py), [`Video`](https://youtu.be/Dm5ptTiIpkk)

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/01/real-time-emotion-mark.png" width="70%" height="70%"></p>

**Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/vgg-face.ipynb), [`Tutorial`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)

Face recognition is mainly based on convolutional neural networks. We feed two face images to a CNN model and it returns a multi-dimensional vector representations. We then compare these representations to determine these two face images are same person or not. 

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/01/face-recognition-demo.png" width="70%" height="70%"></p>

You can find the most popular face recognition models below.

| Model | Creator | LFW Score | Code | Tutorial |
| ---   | --- | --- | ---  | --- |
| VGG-Face | The University of Oxford | 98.78 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/vgg-face.ipynb) | [`Tutorial`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) |
| FaceNet | Google | 99.65 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/vgg-face.ipynb) | [`Tutorial`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) |
| DeepFace | Facebook | - | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Facebook-Deepface.ipynb) | [`Tutorial`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/) |
| OpenFace | Carnegie Mellon University | 93.80 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface.ipynb) | [`Tutorial`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/) |
| DeepID | The Chinese University of Hong Kong | - | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/DeepID.ipynb) | [`Tutorial`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/) |
| Dlib | Davis E. King | 99.38 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Dlib-Face-Recognition.ipynb) | [`Tutorial`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) |
| OpenCV | OpenCV Foundation | - | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/opencv-face-recognition.ipynb) | [`Tutorial`](https://sefiks.com/2020/07/14/a-beginners-guide-to-face-recognition-with-opencv-in-python/) |
| OpenFace in OpenCV | Carnegie Mellon University | 92.92 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/opencv-dnn-face-recognition.ipynb) | [`Tutorial`](https://sefiks.com/2020/07/24/face-recognition-with-opencv-dnn-in-python/) |
| SphereFace | Georgia Institute of Technology | 99.30 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/SphereFace.ipynb) | [`Tutorial`](https://sefiks.com/2020/10/19/face-recognition-with-sphereface-in-python/) |
| ArcFace | Imperial College London | 99.40 | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/ArcFace.ipynb) | [`Tutorial`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) |

All of those state-of-the-art face recognition models are wrapped in [deepface library for python](https://github.com/serengil/deepface). You can build and run them with a few lines of code. To have more information, please visit the [repo](https://github.com/serengil/deepface) of the library.

**Real Time Deep Face Recognition Implementation** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/deep-face-real-time.py), [`Video`](https://www.youtube.com/watch?v=tSU_lNi0gQQ)

These are the real time implementations of the common face recognition models we've mentioned in the previous section. VGG-Face has the highest face recognition score but it comes with the high complexity among models. On the other hand, OpenFace is a pretty model and it has a close accuracy to VGG-Face but its simplicity offers high speed than others.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2020/02/deepface-cover.jpg" width="90%" height="90%"></p>

| Model | Creator | Code | Demo |
| ---   | --- | ---  | --- |
| VGG-Face | Oxford University | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/deep-face-real-time.py) | [`Video`](https://www.youtube.com/watch?v=tSU_lNi0gQQ) |
| FaceNet | Google | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/facenet-real-time.py) | [`Video`](https://youtu.be/vB1I5vWgTQg) |
| DeepFace | Facebook | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/fb-deepface-real-time.py) | [`Video`](https://youtu.be/YjYIMs5ZOfc) |
| OpenFace | Carnegie Mellon University | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/openface-real-time.py) | [`Video`](https://youtu.be/-4z2sL6wzP8) |

**Large Scale Face Recognition**

Face recognition requires to apply face verification several times. It has a O(n) time complexity and it would be problematic for very large scale data sets (millions or billions level data). Herein, if you have a really strong database, then you use relational databases and regular SQL. Besides, you can store facial embeddings in nosql databases. In this way, you can have the power of the map reduce technology. Besides, approximate nearest neighbor (a-nn) algorithm reduces time complexity dramatically. Spotify Annoy, Facebook Faiss and NMSLIB are amazing a-nn libraries. Besides, Elasticsearch wraps NMSLIB and it also offers highly scalablity. You should build and run face recognition models within those a-nn libraries if you have really large scale data sets.

| Library | Algorithm | Tutorial | Code | Demo |
| --- | --- | --- | --- | --- | 
| Spotify Annoy | a-nn | [`Tutorial`](https://sefiks.com/2020/09/16/large-scale-face-recognition-with-spotify-annoy/) | - | [`Video`](https://youtu.be/Jpxm914o2xk) |
| Facebook Faiss | a-nn | [`Tutorial`](https://sefiks.com/2020/09/17/large-scale-face-recognition-with-facebook-faiss/) | - | - |
| NMSLIB | a-nn | [`Tutorial`](https://sefiks.com/2020/09/19/large-scale-face-recognition-with-nmslib/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/nmslib-fast-search.ipynb) | - |
| Elasticsearch | a-nn | [`Tutorial`](https://sefiks.com/2020/11/27/large-scale-face-recognition-with-elasticsearch/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Elastic-Face.ipynb) | [`Video`](https://youtu.be/i4GvuOmzKzo) |
| mongoDB | k-NN | [`Tutorial`](https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/) | [`Code`](https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/) | - |
| Cassandra | k-NN | [`Tutorial`](https://sefiks.com/2021/01/24/deep-face-recognition-with-cassandra/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Cassandra-Face-Recognition.ipynb) | [`Video`](https://youtu.be/VQqHs6-4Ylg) |
| Redis | k-NN | [`Tutorial`](https://sefiks.com/2021/03/02/deep-face-recognition-with-redis/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Redis-Face-Recognition.ipynb) | [`Video`](https://youtu.be/eo-fTv4eYzo) |
| Hadoop | k-NN | [`Tutorial`](https://sefiks.com/2021/01/31/deep-face-recognition-with-hadoop-and-spark/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/PySpark-Face-Recognition.ipynb) | - |
| Relational Database | k-NN | [`Tutorial`](https://sefiks.com/2021/02/06/deep-face-recognition-with-sql/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Face-Recognition-SQL.ipynb) | - |
| Neo4j Graph| k-NN | [`Tutorial`](https://sefiks.com/2021/04/03/deep-face-recognition-with-neo4j/) | [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Neo4j-Face-Recognition.ipynb) | [`Video`](https://youtu.be/X-hB2kBFBXs) |

**Apparent Age and Gender Prediction** [`Tutorial`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`Code for age`](https://github.com/serengil/tensorflow-101/blob/master/python/apparent_age_prediction.ipynb), [`Code for gender`](https://github.com/serengil/tensorflow-101/blob/master/python/gender_prediction.ipynb)

We've used VGG-Face model for apparent age prediction this time. We actually applied transfer learning. Locking the early layers' weights enables to have outcomes fast. 

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/10/age-prediction-for-godfather-original.png" width="70%" height="70%"></p>

We can run age and gender prediction in real time as well [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py), [`Video`](https://youtu.be/tFI7vZn3P7E)

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/02/age-real-time.jpg" width="50%" height="50%"></p>

**Celebrity You Look-Alike Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Find-Look-Alike-Celebrities.ipynb), [`Tutorial`](https://sefiks.com/2019/05/05/celebrity-look-alike-face-recognition-with-deep-learning-in-keras/)

Applying VGG-Face recognition technology for [imdb data set](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) will find your celebrity look-alike if you discard the threshold in similarity score.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/05/sefik-looks-alike-colin-hanks.jpg" width="50%" height="50%"></p>

This can be run in real time as well [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/celebrity-look-alike-real-time.py), [`Video`](https://youtu.be/RMgIKU1H8DY)

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2019/05/celebrity-look-alike-real-time.jpg" width="70%" height="70%"></p>

**Race and Ethnicity Prediction** 
[`Tutorial`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Race-Ethnicity-Prediction-Batch.ipynb), [`Real Time Code`](https://github.com/serengil/tensorflow-101/blob/master/python/real-time-ethnicity-prediction.py), [`Video`](https://youtu.be/-ztiy5eJha8)

Ethnicity is a facial attribute as well and we can predict it from facial photos. We customize VGG-Face and we also applied transfer learning to classify 6 different ethnicity groups.

<p align="center"><img src="https://i0.wp.com/sefiks.com/wp-content/uploads/2019/11/FairFace-testset.png" width="70%" height="70%"></p>

**Beauty Score Prediction** [`Tutorial`](https://sefiks.com/2019/12/25/beauty-score-prediction-with-deep-learning/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Beauty.ipynb)

South China University of Technology published a research paper about facial beauty prediction. They also [open-sourced](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) the data set. 60 labelers scored the beauty of 5500 people. We will build a regressor to find facial beauty score. We will also test the built regressor on a huge imdb data set to find the most beautiful ones.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2020/01/beauty-imdb-v2.png" width="70%" height="70%"></p>

**Attractiveness Score Prediction** [`Tutorial`](https://sefiks.com/2020/01/22/attractiveness-score-prediction-with-deep-learning/), [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/Attracticeness.ipynb)

The University of Chicago open-sourced the Chicago Face Database. The database consists of 1200 facial photos of 600 people. Facial photos are also labeled with attractiveness and babyface scores by hundreds of volunteer markers. So, we've built a machine learning model to generalize attractiveness score based on a facial photo.

<p align="center"><img src="https://sefiks.com/wp-content/uploads/2020/01/attractiveness-cover-2.png" width="70%" height="70%"></p>

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

**Face Alignment for Face Recognition** [`Code`](https://github.com/serengil/tensorflow-101/blob/master/python/face-alignment.py), [`Tutorial`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/)

Google declared that face alignment increase its face recognition model accuracy from 98.87% to 99.63%. This is almost 1% accuracy improvement which means a lot for engineering studies.

<p align="center"><img src="http://sefiks.com/wp-content/uploads/2020/02/rotate-from-scratch.jpg" width="70%" height="70%"></p>

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

There are many ways to support a project - starring⭐️ the GitHub repos is one 🙏

You can also support this work on [Patreon](https://www.patreon.com/serengil?repo=tensorflow101)

<a href="https://www.patreon.com/serengil?repo=tensorflow101">
<img src="https://raw.githubusercontent.com/serengil/tensorflow-101/master/icons/patreon.png" width="30%" height="30%">
</a>

# Citation

Please cite tensorflow-101 in your publications if it helps your research. Here is an example BibTeX entry:

```BibTeX
@misc{serengil2021tensorflow,
  abstract     = {TensorFlow 101: Introduction to Deep Learning for Python Within TensorFlow},
  author       = {Serengil, Sefik Ilkin},
  title        = {tensorflow-101},
  howpublished = {https://github.com/serengil/tensorflow-101},
  year         = {2021}
}
```

# Licence

This repository is licensed under MIT license - see [`LICENSE`](https://github.com/serengil/tensorflow-101/blob/master/LICENSE) for more details
