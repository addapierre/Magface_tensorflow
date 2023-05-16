# Magface_tensorflow

Simple implementation of the face-recognition training method [MagFace](https://arxiv.org/abs/2103.06627) in Tensorflow2, adapted from the author's [PyTorch implementation](https://github.com/IrvingMeng/MagFace/tree/main)
MagFace is a variation of the widely used [ArcFace](https://arxiv.org/abs/1801.07698) for face recognition, with a margin that depends on the feature's magnitude and a "gravity"/regularization parameter, which also depends on the feature magnitude.
With MagFace, the feature norm or magnitude becomes an indicator of the image quality and can be used to select hard images or outliers in a given dataset.


In this file, you can find a custom layer (MagLayer), a custom loss (MagLoss), and a custom Keras model (MagFaceModel).  
This model can be compiled with any optimizer and the custom loss, and then trained and called like any regular tensorflow keras model.  
Calling this model will generate multiple ouput: the classifier output (*cos_theta*), the feature norm (*x_norm*), and the feature itself (*embedding*).  
The feature extraction backbone and the model architecture can be changed in the **\_\_init\_\_** and **call** methods.

A validation callback is also provided, which can be used to monitor the rank-n accuracy and the silhouette score of the validation dataset during training.
