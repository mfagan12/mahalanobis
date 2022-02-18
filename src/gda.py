from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_blobs
import numpy as np


def mahalanobis(samples, mean, covariance):
    shifted = samples - mean
    return np.sqrt(np.diag(shifted @ covariance @ shifted.T))


def batch_mahalanobis_einsum(samples, means, covariances, optimize=True):
    num_samples, dim = samples.shape
    num_clusters = len(means)
    samples = np.expand_dims(samples, axis=1)
    samples = np.tile(samples, (1, num_clusters, 1))
    shifted = samples - means
    # Compute Mahalanobis distance of each point from each cluster. The einsum
    # computes shifted @ covariance @ shifted.T for each cluster efficiently.
    return np.sqrt(np.einsum("ijk,jkl,ijl -> ij", 
                             shifted, covariances, shifted, 
                             optimize=optimize))


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


def naive_batch_mahalanobis(data,
                          means,
                          covariances):
    #! Use the more efficient mahalanobis_einsum
    distances = [mahalanobis(data, mean, covariance) 
                 for mean, covariance 
                 in zip(means, covariances)]
    return np.array(distances).T


def gda_train(data, labels, threshold):
    means, covariances = fit_qda(data, labels)
    distances = train_mahalanobis(data, labels, means, covariances)
    cutoff_dist = distance_at_threshold(distances, threshold)
    return means, covariances, cutoff_dist


def gda_inference(data, means, covariances, cutoff_dist):
    distances = batch_mahalanobis_einsum(data, means, covariances)
    return (distances <= cutoff_dist).any(axis=1)


def fit_qda(data, labels):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(data, labels)
    means = qda.means_
    covariances = qda.covariance_
    return means, covariances


def distance_at_threshold(distances, threshold):
    n = len(distances)
    sorted_dists = sorted(list(distances))
    return sorted_dists[int(n * threshold)]