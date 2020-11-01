import tensorflow as tf

class FullModel(tf.keras.Model):
    def __init__(self, classifier, gda_model, lr, num_layers):
        super(FullModel, self).__init__()
        self.classifier = classifier
        self.gda_model = gda_model
        self.lr = lr
        self.num_layers = num_layers

    def call(self, inputs):
        raise NotImplementedError

if __name__ == "__main__":
    classifier = None
    gda_model = None
    lr = tf.keras.models.load_model("../../models/lr_save")

    full_model = FullModel(classifier, gda_model, lr, 4)

    tf.keras.model.save_model(full_model, "../../model/full_model_save")
