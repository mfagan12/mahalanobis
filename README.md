# Neural network confidence scores from Gaussian discriminant analysis

## By Michael Fagan

This repo is a TensorFlow-based implementation of the techniques in the paper *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* by Lee et al.

First, we will train a neural network to classify the MNIST data set of handwritten digits. Using this classifier, the techniques of the paper will allow us to compute a confidence score, which we hope is more robust than a softmax score, by employing the assumptions of Gaussian discriminant analysis in the intermediate representation spaces of the classifier's layers.
