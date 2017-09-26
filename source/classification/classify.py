# -*- coding: utf-8 -*-
from sklearn import svm

def     classfify_svm(X, y, max_iter = -1):
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, \
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, \
    probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
    clf.fit(X, y)
    return clf  
