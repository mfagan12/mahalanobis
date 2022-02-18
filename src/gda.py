from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_blobs
import numpy as np


def mahalanobis(samples, mean, covariance):
    shifted = samples - mean
    return np.sqrt(np.diag(shifted @ covariance @ shifted.T))


def mahalanobis_einsum(samples, means, covariances):
    # samples: num_examples X dim
    # means: num_clusters X dim
    # covariances: num_clusters, dim, dim
    num_samples, dim = samples.shape
    num_clusters = len(means)
    # samples: num_examples X num_clusters X dim
    samples = np.tile(samples, (num_clusters, 1, 1))
    shifted = samples - means
    return np.einsum("ijk,lkm,nom -> ij", shifted, covariances, shifted)


def make_test_data(n_samples: int = 1000,
                   n_features: int = 2,
                   centers: int = 10):
    data, labels = make_blobs(n_samples=n_samples,
                              n_features=n_features,
                              centers=centers)
    return data, labels


def train_mahalanobis(data,
                     labels,
                     means,
                     covariances,
                     per_class=False):
    distances = np.array([mahalanobis(row.reshape(1,-1),
                             means[label],
                             covariances[label]) 
                          for row, label in zip(data, labels)])
    return distances


def inference_mahalanobis(data,
                          means,
                          covariances):
    distances = [mahalanobis(data, mean, covariance) 
                 for mean, covariance 
                 in zip(means, covariances)]
    return np.array(distances).T


def gda(data, labels):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(data, labels)
    means = qda.means_
    covariances = qda.covariance_
    return means, covariances


def distance_at_threshold(distances, threshold):
    n = len(distances)
    sorted_dists = sorted(distances)
    return sorted_dists[int(n * threshold)]