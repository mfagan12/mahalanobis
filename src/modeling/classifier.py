import tensorflow as tf

class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        raise NotImplementedError

    def layer_out(self, layer_index, inputs):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError
