#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:53:29 2019

@author: michaelyin1994
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from functools import wraps
import pandas as pd
import gc
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering

from UTILS import ReduceMemoryUsage, LoadSave
from UTILS import basic_feature_report, plot_single_record
from UTILS import generating_cutted_sequences

from DistanceMeasurements import dynamic_time_warping
from DistanceMeasurements import fast_dynamic_time_warping
from DistanceMeasurements import get_adjacency_matrix_fastdtw, get_adjacency_matrix_dtw


np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def load_data():
    ls = LoadSave()
    currentData = ls.load_data(path="..//Data//step_1_limited_features.pkl")
    featureData = ls.load_data(path="..//Data//step_2_feature_extraction.pkl")[0]    
    return currentData, featureData
    
def anomaly_detection_stat():
    pass


def clustering_raw_features(features=[], n_clusters=10, method="spectral"):
    if (len(features) == 0) or (n_clusters < 2):
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))
        
    # Step 3: Start clustering using the compressed data
    if method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, n_init=20, gamma=0.5,
                                 affinity="rbf", n_jobs=-1)
    else:    
        clf = KMeans(n_clusters=n_clusters, n_jobs=-1)
    labels = clf.fit_predict(features.values)
    return labels


def clustering_pca_features(features=[], n_clusters=10, n_components=10, method="spectral"):
    if (len(features) == 0) or (n_clusters < 2) or (n_components < 2):
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))
        
    # Step 2: Get the features after the PCA processing
    pca = PCA(n_components=n_components)
    featuresPCA = pca.fit_transform(features.values)
    varRemain = pca.explained_variance_ratio_
    print("Information reamins : {}".format(sum(varRemain)))
    
    # Step 3: Start clustering using the compressed data
    if method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, n_init=20, gamma=0.5,
                                 affinity="rbf", n_jobs=-1)
    else:    
        clf = KMeans(n_clusters=n_clusters, n_jobs=-1)
    labels = clf.fit_predict(features.values)
    labels = clf.fit_predict(featuresPCA)
    return labels


def clustering_dtw_spectral(data=[], n_clusters=10):
    
    pass


def clustering_fastdtw_spectral(data=[], n_clusters=10):
    
    pass


def clustering_dae():
    pass


def clustering_ae():
    pass


def plot_clustering_results(data=[], c_label=[], t_label=[], methodStr="..//Plots//FeaturePcaBased//FeaturePcaBased_", plotNums=100):
    '''
    1. Generating clustering reports
    2. Plot each cluster(Pick certain number of signal), plot them
    '''
    if (len(data) == 0) or (len(data) != len(c_label)):
        return None
    
    # Accessing the unique cluster labels of the data
    uniqueLabels = np.unique(c_label)

    # Plot the current data
    for label in uniqueLabels:
        indexList = list(data[c_label == label].index)
        
        # Randomly select the number of plotNums sequences.
        plotList = []
        for i in range(plotNums):
            plotList.append(random.choice(indexList))
        
        # Start plot
        f, ax = plt.subplots()
        for i in plotList:
            signal = data.iloc[i]["current"]
            plt.plot(signal, color="blue", linewidth=1.2, linestyle="-")
        plt.xlabel("time(s)")
        plt.ylabel("current(A)")
        plt.title("Cluster " + str(label))
        plt.savefig(methodStr + str(label) + ".png", dpi=500, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    # Loading the data
    currentData, featureData = load_data()
    res = currentData["current"].values
    adjacencyMat = get_adjacency_matrix_fastdtw(res, verboseRound=10, radius=1)
    
#    featurePcaLabels = clustering_pca_features(featureData.drop(["phase", "dateTime", "record"], axis=1).copy(),
#                                               n_components=int(0.8 * featureData.shape[1]), n_clusters=6, method="KMeans")
#    featureLabels = clustering_raw_features(featureData.drop(["phase", "dateTime", "record"], axis=1).copy(),
#                                            n_clusters=6, method="KMeans")
    
#    plot_clustering_results(currentData, featurePcaLabels, plotNums=50, methodStr="..//Plots//FeaturePcaBased//FeaturePcaBased_")
#    plot_clustering_results(currentData, featurePcaLabels, plotNums=50, methodStr="..//Plots//RawFeatureBased//RawFeatureBased_")
    ret = generating_cutted_sequences(currentData["current"].values, window=60, stride=20, portion=0.3, verboseRound=5000)