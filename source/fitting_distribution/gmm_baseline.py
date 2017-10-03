import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
import pickle

import sys
sys.path.append(".")



class  Cluster:
    def __init__(self, mean, cov, weight = 1):
        self.mean = mean
        self.cov = cov
        self.weight = weight
        
#public
def cal_logp_item_cluster(feature_item, cluster):
    return multivariate_normal.logpdf(feature_item, mean = cluster.mean, cov = cluster.cov)

#not public
def cal_logp_user_cluster(user, features, cluster):
    return np.sum([cal_logp_item_cluster(features[item], cluster) for item in user])

#public
def cal_ptrans_user(user, features, clusters):
    return np.sum([cluster.weight * cal_logp_user_cluster(user, features, cluster) for cluster in clusters])

#public
def cal_ptrans_item_user(item, user, features, clusters):
    return np.sum([cal_logp_item_cluster(features[item], cluster) + cal_logp_user_cluster(user, features, cluster)\
                   for cluster in clusters])
    
def  find_clusters(items, features, n_cluster):
    '''
        itemset: list indexes of images corresponding to video_img. Discard indexes where feature is None.
        usr_list: |U| users, each user is a sets of items.
        feature: gist or lab feature coressponding image in video_img.
    '''
    label_init = np.random.randint(0, n_cluster, len(items))

    gmm = GaussianMixture(n_components=n_cluster, covariance_type = 'full', max_iter = 100, random_state=0)
    gmm.means_init = np.array([features[items[label_init == i]].mean(axis=0) for i in range(n_cluster)])
    gmm.fit(features[items])
    return gmm


def  validate(gmm, usr_list, features):
    means, covs, weights = gmm.means_, gmm.covariances_, gmm.weights_
    clusters = [Cluster(means[i], covs[i], weights[i]) for i in range(len(means))]
    
    p_users = []
    for usr in usr_list:
        p_user = cal_ptrans_user(usr, features, clusters)
        p_users.append(p_user)
        print p_user
    return np.array(p_users).mean()


usr_list = None
with open ('../../data/usr_his_sample/usr_list_5-20.txt', 'rb') as fp:
    usr_list = pickle.load(fp) 
len(usr_list)

features = np.loadtxt('../../data/usr_his_sample/features.txt')

item_list = np.loadtxt('../../data/usr_his_sample/item_list.txt').astype(int)

gmm = find_clusters(item_list, features, n_cluster=32)

means, covs, weights = gmm.means_, gmm.covariances_, gmm.weights_
clusters = [Cluster(means[i], covs[i], weights[i]) for i in range(len(means))]

