import tensorflow as tf
from tf.keras.layers import *

class GDALayer(tf.keras.Layer):
    def __init__(self):
        raise NotImplementedError

    def _mahalanobis(data, mean, precison):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

class LogisticRegressor(tf.keras.Model):
    def __init__(self)
        super(LogisticRegressor, self).__init__()
        self.dense = Dense(1, activation = "sigmoid")

    def call(self, inputs):
        return self.dense(inputs)
