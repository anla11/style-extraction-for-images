import numpy as np
from sklearn.externals import joblib

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

def     kmeans_predict(kmeans, X, n_clusters):
    predictions = kmeans.predict(X)
    clusters = []
    for i in range(n_clusters):
        cluster_i = np.array(np.where(predictions == i)[0])
        clusters.append(cluster_i)
    return clusters    

def     kmeans_cal(X, n_clusters):
    return KMeans(n_clusters = n_clusters, random_state=0).fit(X)

def     get_kmeans(features, n_clusters, cluster_model, mode = 'test'):
    kmeans = None
    if mode == 'train':
        kmeans = kmeans_cal(features, n_clusters)
    else:
        print 'Load model %s' % cluster_model
        kmeans = joblib.load(cluster_model)    
    clusters = kmeans_predict(kmeans, features, n_clusters)        
    return kmeans, clusters# -*- coding: utf-8 -*-


def cal_position(dab, dac, dbc):
    x = (dac*dac + dab*dab - dbc*dbc)/(2*dab)
    if dac == 0: 
        y = 0
    else:
        y = np.sqrt(dac*dac - x*x)
    return x, y    

def cal_distance(features):
    f0 = np.zeros_like(features[0])
    f1 = np.ones_like(features[0])
    d = euclidean(f0, f1) / np.sqrt(len(f1))
    pos = []
    for f in features:
        d0 = euclidean(f0, f) / np.sqrt(len(f1))
        d1 = euclidean(f1, f) / np.sqrt(len(f1))
        x, y = cal_position(d, d0, d1)
        pos.append((x, y))
    return pos