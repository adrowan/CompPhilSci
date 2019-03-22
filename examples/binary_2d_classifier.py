#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CompPhilSci: Computational Philosophy of Science Library"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier


cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


def plot_binary_classifier_probabilities(classifier, X, y, y_pred, title='', hyperplane_col='grey', plot_colmesh=True):
    
    tp = (y == y_pred)  # tp = true positive; fp = false positive.
    tp0, tp1 = tp[y == 0], tp[y == 1]  # true positives for classes zero and one 
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0] # true and false positives 
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    
    plt.title(title)
    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')  # plot class zero (Red) true positives
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x', s=20, color='#990000')  # plot class zero (Red) false positives (marking them as dark red crosses)
    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue') # plot class one (Blue) true positives
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x', s=20, color='#000099') # plot class zero (Red) false positives (marking them as dark blue crosses)
    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim() # limits taken from scatter plot above
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)) # returns a list of 2 elements: each grid corresponding to 1st and 2nd arguments respectively
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()]) # pass (n_samples, n_features), get back (n_samples, n_classes)
    Z = Z[:, 1].reshape(xx.shape)
    if plot_colmesh==True:
        plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], colors=hyperplane_col) #linewidths=1.
    
    plt.show()



np.random.seed(3)
X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=2.0)
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) 


lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
y_train_pred_lda = lda.fit(X_train, y_train).predict(X_train)
y_test_pred_lda  = lda.fit(X_train, y_train).predict(X_test)

plot_binary_classifier_probabilities(lda, X_train, y_train, y_train_pred_lda, title='Linear Discriminant; Training Set Accommodation', hyperplane_col='grey', plot_colmesh=True)
plot_binary_classifier_probabilities(lda, X_test , y_test , y_test_pred_lda , title='Linear Discriminant; Test Set Prediction'    , hyperplane_col='grey', plot_colmesh=True)


svc = svm.SVC(kernel='rbf', C=1.0, probability=True)
y_train_pred_svc = svc.fit(X_train, y_train).predict(X_train)
y_test_pred_svc  = svc.fit(X_train, y_train).predict(X_test)

plot_binary_classifier_probabilities(svc, X_train, y_train, y_train_pred_svc, title='Support Vector Machine; Training Set Accommodation', hyperplane_col='grey', plot_colmesh=True)
plot_binary_classifier_probabilities(svc, X_test , y_test , y_test_pred_svc , title='Support Vector Machine; Test Set Prediction'    , hyperplane_col='grey', plot_colmesh=True)


mlp = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(100, 100), alpha=0.005, max_iter=400, verbose=False, random_state=1) # 2 x 100-wide layers
y_train_pred_mlp = mlp.fit(X_train, y_train).predict(X_train)
y_test_pred_mlp  = mlp.fit(X_train, y_train).predict(X_test)

plot_binary_classifier_probabilities(mlp, X_train, y_train, y_train_pred_mlp, title='Neural Network; Training Set Accommodation', hyperplane_col='grey', plot_colmesh=True)
plot_binary_classifier_probabilities(mlp, X_test , y_test , y_test_pred_mlp , title='Neural Network; Test Set Prediction'    , hyperplane_col='grey', plot_colmesh=True)
