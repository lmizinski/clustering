from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
#from time import time

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

# Define the number of inliers and outliers
#n_samples = 200
outliers_fraction = 0.25
#clusters_separation = [0]

# Compare given detectors under given settings
# Initialize the data
#xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
#n_inliers = int((1. - outliers_fraction) * n_samples)
#n_outliers = int(outliers_fraction * n_samples)
#ground_truth = np.zeros(n_samples, dtype=int)
#ground_truth[-n_outliers:] = 1

# Show the statics of the data
#print('Number of inliers: %i' % n_inliers)
#print('Number of outliers: %i' % n_outliers)
#print('Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format(shape=ground_truth.shape))
#print(ground_truth)

random_state = np.random.RandomState(42)
# Define nine outlier detection tools to be compared
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(n_neighbors=10,contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction, check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                      contamination=outliers_fraction,
                      check_estimator=False,
                      random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': 
        HBOS(contamination=outliers_fraction),
    'Isolation Forest': 
        IForest(contamination=outliers_fraction,random_state=random_state),
    'K Nearest Neighbors (KNN)': 
        KNN(contamination=outliers_fraction),
    'Average KNN': 
        KNN(method='mean', contamination=outliers_fraction),
    'Median KNN': 
        KNN(method='median', contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
       LOF(n_neighbors=35, contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': 
        MCD(contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': 
        OCSVM(contamination=outliers_fraction, random_state=random_state),
    'Principal Component Analysis (PCA)': 
        PCA(contamination=outliers_fraction, random_state=random_state),
}

# Show all detectors
#for i, clf in enumerate(classifiers.keys()):
#    print('Model', i + 1, clf)



#offset = 0;
#np.random.seed(42)
# Data generation
#X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
#X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
#X = np.r_[X1, X2]
# Add outliers
#X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

def get_csv_data(data_limit=None):
    fueldata = [];
    i = 0;
    with open('fueldataset2.csv', newline="\r\n") as csvfile:
        spamreader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        for row in spamreader:
            if row and row[0]!='frame_order':
                new_element = [int(row[0]),int(row[2])] #0-frameorder, 1-sum, 2-raw, 3-gpsdatetime
                fueldata.append(new_element)
    if data_limit is None:
        result = fueldata
    else:
        result = fueldata[0::data_limit]
    return np.asarray(result);

X = get_csv_data(10)
X = np.array(X)
X_len = len(X)

# Fit the models with the generated data and compare model performances
plt.figure(figsize=(15, 12))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    t0 = time.time()
    print(i + 1, 'fitting', clf_name)
    # fit the data and tag outliers
    clf.fit(X)
    
    scores_pred = clf.decision_function(X) * -1
    y_pred = clf.predict(X)
    
    outliers = [];
    for key, val in enumerate(X):
        if (y_pred[key] == -1 or y_pred[key] == 1):
            outliers.append(val);

    outliers = np.array(outliers)
    outliers_len = len(outliers)
    t1 = time.time()
    #print(X,y_pred,outliers)
    #sys.exit()
    subplot = plt.subplot(3, 4, i + 1)

    
    b = subplot.scatter(X[:, 0], X[:, 1], c='g', s=4, edgecolor='')
    if outliers_len>0:
        c = subplot.scatter(outliers[:, 0], outliers[:, 1], c='r', s=4, edgecolor='')
    else:
        c = []
        
    subplot.axis('tight')
    subplot.legend(
        [b, c],
        ["all data (%d)" % X_len, 'outliers (%d)' % outliers_len],
        prop=matplotlib.font_manager.FontProperties(size=8),
        loc='lower right')
    subplot.set_xlabel("%d. %s" % (i + 1, clf_name))
    subplot.text(.15, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    #subplot.set_xlim((-7, 7))
    #subplot.set_ylim((-7, 7))
plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
plt.suptitle("Outlier detection")
plt.show()
