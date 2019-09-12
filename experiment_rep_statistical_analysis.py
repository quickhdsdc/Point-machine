#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:24:06 2019

@author: prometheus
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import seaborn as sns
from functools import wraps
import pandas as pd
import gc
from datetime import datetime
import random
from scipy import linalg
import itertools
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score

from UTILS import LoadSave
from UTILS import lgb_clf_training, plot_feature_importance

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["C" + str(i) for i in range(0, 9+1)]
mpl.rcParams["font.family"] = "Times New Roman"
###############################################################################
###############################################################################
def load_data():
    fileName = ["current_ts_merged_3300.pkl"]
    ls = LoadSave()
    currentData = ls.load_data(".//Data//" + fileName[0])
    return currentData


def select_best_lof_value(data=None, y_true=None):
    rocRecord, nnVal = [], [i for i in range(5, 250, 3)]
    socres = []
    
    for ind, item in enumerate(nnVal):
        clf = LocalOutlierFactor(n_neighbors=item, contamination=0.05,
                                 metric="l2", n_jobs=1)
        clf.fit(data)
        noiseLevel = clf.negative_outlier_factor_
        fpr, tpr, _ = roc_curve(y_true, noiseLevel)
        rocRecord.append(auc(fpr, tpr))
        socres.append([noiseLevel, fpr, tpr])
        
    return np.argmax(rocRecord), max(rocRecord), [nnVal, rocRecord, socres]


def feature_based_anomaly_detection(features=[], n_components=10):
    if len(features) == 0:
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True, axis=0)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))

    # Step 2: Get the features after the PCA processing
    pca = PCA(n_components=n_components)
    featuresPCA = pca.fit_transform(features.values)
    varRemain = pca.explained_variance_ratio_
    print("@Information reamins : {}".format(sum(varRemain)))
    
    return featuresPCA
###############################################################################
###############################################################################
if __name__ == "__main__":
#    currentData = load_data()
#    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
#    
#    signal = {"current": ts}
#    signal = pd.DataFrame(signal, columns=["current"])
#    plotSave, plotShow = True, False
#    ###############################################
#    ###############################################
#    '''
#    Step 1: Load all DAE representations, and perform the anamoly detection.
#    '''
#    PATH = ".//Data//rep_data//"
#    ls = LoadSave(PATH)
#    fileName = os.listdir(PATH)[:4]
#    
#    # Get the representations
#    featureRep = feature_based_anomaly_detection(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
#    rep = []
#    for name in fileName:
#        rep.append(ls.load_data(PATH + name))
#    
#    # Anamoly detection: select the best score
#    featureRocBestInd, featureRocBest, featureRocRec = select_best_lof_value(data=featureRep,
#                                                                             y_true=np.where(groundTruth["label"] != -1, 1, -1))
#    rocBestInd, rocBest, rocRec = [], [], []
#    for data_rep in rep:
#        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep,
#                                                    y_true=np.where(groundTruth["label"] != -1, 1, -1))
#        rocBestInd.append(tmp_1)
#        rocBest.append(tmp_2)
#        rocRec.append(tmp_3)
    ###############################################
    ###############################################
    '''
    Plot 1: DIfferent cluster and its representations.
    '''
    fig, axObj = plt.subplots(len(rep)+1, 4, figsize=(12, 10))
    labelToPlot, signalsEachPlot = [1, 2, 3, -1], 5
    selectedSamples = []
    for label in labelToPlot:
        sampleInd = groundTruth[groundTruth["label"] == label]["ind"].values
        selectedSamples.append(np.random.choice(sampleInd, size=signalsEachPlot, replace=False))
    
    # Plot the original signals
    for ind, item in enumerate(axObj[0]):
        sampleInd = selectedSamples[ind]
        for i in sampleInd:
            item.plot(ts[int(i)], color="k", linewidth=1.5)
            item.tick_params(axis="y", labelsize=8)
            item.tick_params(axis="x", labelsize=8)
            
    # Plot each cluster of the signals
    for ax_ind in range(1, len(rep)+1):
        for ind, item in enumerate(axObj[ax_ind]):
            sampleInd = selectedSamples[ind]
            for i in sampleInd:
                item.plot(rep[ax_ind-1][int(i), :], color="b", linewidth=1.5)
                item.tick_params(axis="y", labelsize=8)
                item.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    
    if plotSave == True:
        plt.savefig(".//Plots//different_rep_with_different_cluster.png", dpi=500, bbox_inches="tight")
    ###############################################
    ###############################################
#    '''
#    Plot 2: Hidden representation of the rep with the-same-dim.
#    '''
#    fig, axObj = plt.subplots(2, 2, figsize=(12, 8))
#    
#    for ind, ax in enumerate(axObj.ravel()):
#        tmp = pd.DataFrame(rep[ind])
#        featureCorr = tmp.corr()
#        ax = sns.heatmap(featureCorr, xticklabels=2, yticklabels=2,
#                         cmap="Blues", fmt='.2f', annot=True,
#                         annot_kws={'size':5.5,'weight':'bold'}, 
#                         vmin=-1, vmax=1, ax=ax)
#        ax.tick_params(axis="y", labelsize=8)
#        ax.tick_params(axis="x", labelsize=8)
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=8)
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//rep_corr_plot.png", dpi=500, bbox_inches="tight")
#    plt.close("all")
    ###############################################
    ###############################################
    '''
    Plot 3: Dimen-redunction for the rep.
    '''
#    fig, axObj = plt.subplots(2, 2, figsize=(12, 8))
#    for i, ax in enumerate(axObj.ravel()):
#        
#        X_sc = StandardScaler()
#        rep_pca = X_sc.fit_transform(rep[i])
#        
#        clf = KernelPCA(n_components=2, kernel="rbf", gamma=0.05)
#        rep_pca = clf.fit_transform(rep_pca)
#        
#        uniqueLabels = -np.sort(-groundTruth["label"].unique())
#        for j, label in enumerate(uniqueLabels):
#            sampleIndex = np.arange(0, len(groundTruth))
#            labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
#            
#            coords = rep_pca[labeledSampleIndex, :]
#            if label != -1:
#                ax.scatter(coords[:, 0], coords[:, 1], s=10, 
#                           color=colors[j], marker=".", label="Class " + str(j))
#            else:
#                ax.scatter(coords[:, 0], coords[:, 1], s=10, 
#                           color='r', marker="x", label="Class Abnormal")
#            ax.tick_params(axis="y", labelsize=8)
#            ax.tick_params(axis="x", labelsize=8)
#            ax.legend(fontsize=8)
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//different_rep_anomaly.png", dpi=500, bbox_inches="tight")
#    plt.close("all")    


