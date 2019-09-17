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
    rocRecord, nnVal = [], [i for i in range(100, 200, 3)]
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
    currentData = load_data()
    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
    
    signal = {"current": ts}
    signal = pd.DataFrame(signal, columns=["current"])
    plotSave, plotShow = True, False
    ###############################################
    ###############################################
    '''
    Step 1: Load all DAE representations, and perform the anamoly detection.
    '''
    PATH = ".//Data//rep_data//"
    ls = LoadSave(PATH)
    fileName = os.listdir(PATH)
    
    # Get the representations
    featureRep = feature_based_anomaly_detection(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
    rep = []
    for name in fileName:
        rep.append(ls.load_data(PATH + name))
    
    # Anamoly detection: select the best score
#    featureRocBestInd, featureRocBest, featureRocRec = select_best_lof_value(data=featureRep,
#                                                                             y_true=np.where(groundTruth["label"] != -1, 1, -1))
#    rocBestInd, rocBest, rocRec = [], [], []
#    print("\n@Detecting the anamoly data")
#    print("=============================================")
#    for ind, data_rep in enumerate(rep):
#        print("    --Current rep is {}, total is {}.".format(ind+1, len(rep)))
#        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep,
#                                                    y_true=np.where(groundTruth["label"] != -1, 1, -1))
#        rocBestInd.append(tmp_1)
#        rocBest.append(tmp_2)
#        rocRec.append(tmp_3)
#    print("=============================================")
    ###############################################
    ###############################################
    '''
    Plot 1: Different cluster and its representations.
    '''
#    fig, axObj = plt.subplots(len(rep)+1, 4, figsize=(12, 5))
#    labelToPlot, signalsEachPlot = [1, 2, 3, -1], 10
#    selectedSamples = []
#    for label in labelToPlot:
#        sampleInd = groundTruth[groundTruth["label"] == label]["ind"].values
#        selectedSamples.append(np.random.choice(sampleInd, size=signalsEachPlot, replace=False))
#    
#    # Plot the original signals
#    for ind, item in enumerate(axObj[0]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(ts[int(i)], color="k", linewidth=1.5)
#            item.tick_params(axis="y", labelsize=8)
#            item.tick_params(axis="x", labelsize=8)
#            
#    # Plot each cluster of the signals
#    for ax_ind in range(1, len(rep)+1):
#        for ind, item in enumerate(axObj[ax_ind]):
#            sampleInd = selectedSamples[ind]
#            for i in sampleInd:
#                item.plot(rep[ax_ind-1][int(i), :], color="b", linewidth=1.5)
#                item.tick_params(axis="y", labelsize=8)
#                item.tick_params(axis="x", labelsize=8)
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//2_REPANA_different_rep_of_different_cluster.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
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
#    fig, axObj = plt.subplots(1, 2, figsize=(12, 8))
#    for i, ax in enumerate(axObj.ravel()):
#        
#        X_sc = StandardScaler()
#        rep_pca = X_sc.fit_transform(rep[i])
#        
#        clf = KernelPCA(n_components=2, kernel="rbf", gamma=0.05)
#        rep_pca = clf.fit_transform(rep_pca)
#        
#        uniqueLabels = [1, 2, 3, -1]
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
#        plt.savefig(".//Plots//2_REPANA_different_rep_anomaly.png", dpi=500, bbox_inches="tight")
#    plt.close("all")


    '''
    Reperesentation plot(Randomly select 4 classes, then plot 10 signals of
                          each classes with their representations.)
    '''
#    fig, axObj = plt.subplots(6, 5, figsize=(16, 12))
#    labelToPlot, plotNums = [2, 4, 6, 8, -1], 5
#    selectedSamples = []
#    for label in labelToPlot:
#        sampleInd = groundTruth[groundTruth["label"] == label]["ind"].values
#        selectedSamples.append(np.random.choice(sampleInd, size=plotNums, replace=False))
#    
#    for ind, item in enumerate(axObj[0]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(ts[int(i)], color="k", linewidth=1.5)
#            
#    
#    for ind, item in enumerate(axObj[1]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(rep_0[int(i), :], color="b", linewidth=1.5)
#
#    for ind, item in enumerate(axObj[2]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(rep_1[int(i), :], color="b", linewidth=1.5)
#
#    for ind, item in enumerate(axObj[3]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(rep_2[int(i), :], color="b", linewidth=1.5)
#
#    for ind, item in enumerate(axObj[4]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(rep_3[int(i), :], color="b", linewidth=1.5)   
#
#    for ind, item in enumerate(axObj[5]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(rep_4[int(i), :], color="b", linewidth=1.5) 
#    
#    axObj[0][0].set_ylabel("Current(A)", fontsize=12)
#    axObj[1][0].set_ylabel("DAE_0")
#    axObj[2][0].set_ylabel("DAE_1")
#    axObj[3][0].set_ylabel("DAE_2")
#    axObj[4][0].set_ylabel("DAE_3")
#    axObj[5][0].set_ylabel("DAE_4")        
#    plt.tight_layout()
    
    '''
    Classification error plot.(Deep network and sharrow network)
    '''
#    plt.figure(figsize=(9, 9))
#    plt.plot(rocRec_0[1], featureRocRec[1], marker="s", markersize=6,
#             color="b", linestyle=" ", label="DAE_0")
#    plt.plot(rocRec_1[1], featureRocRec[1], marker="o", markersize=6,
#             color="r", linestyle=" ", label="DAE_1")
#    plt.plot(rocRec_2[1], featureRocRec[1], marker="^", markersize=6,
#             color="k", linestyle=" ", label="DAE_2")
#    plt.plot(rocRec_3[1], featureRocRec[1], marker="*", markersize=6,
#             color="g", linestyle=" ", label="DAE_3")
#    plt.xlabel("ROC of DAEs")
#    plt.ylabel("ROC of FeatureBased")
#    plt.legend()
#    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#    plt.xlim(min(min(rocRec_1[0]), min(featureRocRec[1])), 0.91)
#    plt.ylim(min(min(rocRec_1[0]), min(featureRocRec[1])), 0.91)
    
    '''
    Feature importances plot.
    '''
#    lgbParams = {'n_estimators': 2500,
#                 'objective': 'binary',
#                 'boosting_type': 'gbdt',
#                 'n_jobs': -1,
#                 'learning_rate': 0.048,
#                 'num_leaves': 255, # important
#                 'max_depth': 8, # important
#                 'subsample': 0.80,
#                 'subsample_freq': 1,
#                 'colsample_bytree':0.80,
#                 'reg_alpha': 0.16154,
#                 'reg_lambda': 0.8735294,
#                 'silent': 1, 
#                 'max_bin': 64}
#    X_train, X_test = pd.DataFrame(rep_0), pd.DataFrame(rep_0)
#    X_train["target"] = np.where(groundTruth["label"] != -1, 0, 1)
#    lgbScore, lgbImportances, lgbPred = lgb_clf_training(X_train, X_test, lgbParams=lgbParams, numFolds=5, stratified=True)
#    plot_feature_importance(lgbImportances)
#
#    '''
#    Rep_0 corrlation plot.
#    '''
#    dataCorr = pd.DataFrame(rep_0).corr()
#    f, ax = plt.subplots(figsize=(16, 9))
#    sns.heatmap(dataCorr, cmap="Blues", fmt='.2f', annot=True, annot_kws={'size':7.5,'weight':'bold'}, ax=ax)
#    
#    # Weights heatplot
#    fig, axObj = plt.subplots()
#    sns.heatmap(model.layers[1].get_weights()[0].T, cmap="Blues", ax=axObj)