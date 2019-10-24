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

from UTILS import LoadSave, weighting_rep, rf_clf_training, plot_feature_importance
from StackedDenoisingAE import StackedDenoisingAE

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["C" + str(i) for i in range(0, 9+1)]
markers = ["s", "^", "o", "d", "*", "<", ">", "h", "p"]
legends = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
mpl.rcParams["font.family"] = "Times New Roman"
###############################################################################
###############################################################################
def load_data():
    fileName = ["current_ts_merged_3300.pkl"]
    ls = LoadSave()
    currentData = ls.load_data(".//Data//" + fileName[0])
    return currentData


def select_best_lof_value(data=None, y_true=None, nn_range=[i for i in range(60, 300, 5)]):
    rocRecord = []
    socres = []
    
    for ind, item in enumerate(nn_range):
        clf = LocalOutlierFactor(n_neighbors=item, contamination=0.05,
                                 metric="l2", n_jobs=1)
        clf.fit(data)
        noiseLevel = clf.negative_outlier_factor_
        fpr, tpr, _ = roc_curve(y_true, noiseLevel)
        rocRecord.append(auc(fpr, tpr))
        socres.append([noiseLevel, fpr, tpr])
        
    return np.argmax(rocRecord), max(rocRecord), [nn_range, rocRecord, socres]


def feature_kernel_pca(features=[], n_components=10):
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
    y_true = np.where(groundTruth["label"] != -1, 1, -1)
    
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
    fileNameAE = sorted([name for name in fileName if ("rnn" not in name) and ("base" not in name)])
    fileNameRNN = sorted([name for name in fileName if "rnn" in name])
    fileNameBase = sorted([name for name in fileName if "base" in name])
    
    # Get the representations
    load_all = True
    if load_all:
        featureRep = feature_kernel_pca(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
        rep, repOriginal, repRNN, repBase = [], [], [], []
        for name in fileNameAE:
            df = ls.load_data(PATH + name)
            repOriginal.append(df.copy())
            rep.append(weighting_rep(df, norm_factor=0.1, bin_name="bin_freq_10"))
            
        for name in fileNameRNN:
            repRNN.append(ls.load_data(PATH + name))
        for name in fileNameBase:
            repBase.append(ls.load_data(PATH + name))
            repBase[-1] = repBase[-1][[name for name in repBase[-1].columns if "rep" in name]].values
            
    ###############################################
    ###############################################
    # Anamoly detection: select the best score
    # Step ==> 10, 20, 30, neurons(15), windowSize(40)(Bset visualizing)
    #----------------------------------------------------
    repList = [rep[18][0], rep[19][0], repBase[0], repRNN[0],
               rep[18][1], rep[19][1], repBase[1], repRNN[1]]

    ###############################################
    ###############################################
    '''
    Plot 1: Different cluster and its representations.
    '''
#    fig, axObj = plt.subplots(len(repList)+1, 4, figsize=(16, 12))
#    labelToPlot, signalsEachPlot = [1, 2, 3, -1], 100
#    selectedSamples = []
#    for label in labelToPlot:
#        sampleInd = groundTruth[groundTruth["label"] == label]["ind"].values
#        selectedSamples.append(np.random.choice(sampleInd, size=signalsEachPlot, replace=False))
#    
#    lw = 0.8
#    # Plot the original signals
#    for ind, item in enumerate(axObj[0]):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(ts[int(i)], color="k", linewidth=lw)
#            item.tick_params(axis="y", labelsize=8)
#            item.tick_params(axis="x", labelsize=8)
#            
#    # Plot each cluster of the signals
#    for ax_ind in range(1, len(repList)+1):
#        for ind, item in enumerate(axObj[ax_ind]):
#            sampleInd = selectedSamples[ind]
#            for i in sampleInd:
#                item.plot(repList[ax_ind-1][int(i), :], color="b", linewidth=lw)
#                item.tick_params(axis="y", labelsize=8)
#                item.tick_params(axis="x", labelsize=8)
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//2_REPANA_different_rep_of_different_cluster.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
    ###############################################
    ###############################################
    '''
    Plot 2: Hidden representation of the rep with the-same-dim.
    '''
#    fig, axObj = plt.subplots(2, 4, figsize=(16, 5))
#    
#    for ind, ax in enumerate(axObj.ravel()):
#        tmp = pd.DataFrame(repList[ind])
#        featureCorr = tmp.corr()
##        ax = sns.heatmap(featureCorr, xticklabels=2, yticklabels=2,
##                         cmap="Blues", fmt='.2f', annot=True,
##                         annot_kws={'size':3.5,'weight':'bold'}, ax=ax)
#        ax = sns.heatmap(featureCorr, cmap="Blues", ax=ax)
#        ax.tick_params(axis="y", labelsize=8)
#        ax.tick_params(axis="x", labelsize=8)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=8)
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//2_REPANA_rep_corr_plot.pdf", dpi=500, bbox_inches="tight")
#    plt.close("all")
    ###############################################
    ###############################################
    '''
    Plot 3: Dimen-redunction for the rep.
    '''
    
    '''
    Plot the baseline representations.
    '''
    featureTmp = features.copy()
    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
    repList = [featureTmp, repBase[0], repRNN[0],
               featureTmp, repBase[1], repRNN[1]]
    
    fig, axObj = plt.subplots(2, 3, figsize=(14, 7))
    for i, (ax, legend) in enumerate(zip(axObj.ravel(), legends)):
        
        X_sc = StandardScaler()
        rep_pca = X_sc.fit_transform(repList[i])
        
        if i != 3:
            clf = KernelPCA(n_components=2, kernel="rbf")
            rep_pca = clf.fit_transform(rep_pca)
        else:
            clf = PCA(n_components=2)
            rep_pca = clf.fit_transform(rep_pca)
        
        uniqueLabels = [1, 2, 3, -1]
        for j, label in enumerate(uniqueLabels):
            sampleIndex = np.arange(0, len(groundTruth))
            labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
            
            coords = rep_pca[labeledSampleIndex, :]
            if label != -1:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4, 
                           color=colors[j], marker=markers[j], label="Class " + str(label))
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4,
                           color='r', marker="x", label="Class -1")
            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=10)
            ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel(legend, fontsize=10)
    plt.tight_layout()
    
    if plotSave == True:
        plt.savefig(".//Plots//2_REPANA_basline_representations.pdf", dpi=500, bbox_inches="tight")
    plt.close("all")
    
    '''
    Plot the proposed unweighted representations.
    '''
    repList = [rep[12][0], rep[13][0], rep[17][0], 
               rep[14][0], rep[15][0], rep[16][0]]
    
    fig, axObj = plt.subplots(2, 3, figsize=(14, 7))
    for i, (ax, legend) in enumerate(zip(axObj.ravel(), legends)):
        
        X_sc = StandardScaler()
        rep_pca = X_sc.fit_transform(repList[i])
        
        clf = KernelPCA(n_components=2, kernel="rbf")
        rep_pca = clf.fit_transform(rep_pca)
        
        uniqueLabels = [1, 2, 3, -1]
        for j, label in enumerate(uniqueLabels):
            sampleIndex = np.arange(0, len(groundTruth))
            labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
            
            coords = rep_pca[labeledSampleIndex, :]
            if label != -1:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4, 
                           color=colors[j], marker=markers[j], label="Class " + str(label))
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4,
                           color='r', marker="x", label="Class -1")
            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=10)
            ax.legend(fontsize=9, loc="best")
        ax.set_xlabel(legend, fontsize=10)
    plt.tight_layout()
    
    if plotSave == True:
        plt.savefig(".//Plots//2_REPANA_proposed_representations_simple_average.pdf", dpi=500, bbox_inches="tight")
    plt.close("all")
    
    '''
    Plot the proposed weighted representations.
    '''
    repList = [rep[12][1], rep[13][1], rep[17][1], 
               rep[14][1], rep[15][1], rep[16][1]]
    
    fig, axObj = plt.subplots(2, 3, figsize=(14, 7))
    for i, (ax, legend) in enumerate(zip(axObj.ravel(), legends)):
        
        X_sc = StandardScaler()
        rep_pca = X_sc.fit_transform(repList[i])
        
        clf = KernelPCA(n_components=2, kernel="rbf")
        rep_pca = clf.fit_transform(rep_pca)
        
        uniqueLabels = [1, 2, 3, -1]
        for j, label in enumerate(uniqueLabels):
            sampleIndex = np.arange(0, len(groundTruth))
            labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
            
            coords = rep_pca[labeledSampleIndex, :]
            if label != -1:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4, 
                           color=colors[j], marker=markers[j], label="Class " + str(label))
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=9, alpha=0.4,
                           color='r', marker="x", label="Class -1")
            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=10)
            if i == 0:
                ax.legend(fontsize=9, loc="upper right")
            else:
                ax.legend(fontsize=9, loc="best")
        ax.set_xlabel(legend, fontsize=10)
    plt.tight_layout()
    
    if plotSave == True:
        plt.savefig(".//Plots//2_REPANA_proposed_representations_weighted.pdf", dpi=500, bbox_inches="tight")
    plt.close("all")

    ###############################################
    ###############################################
    '''
    Feature importances plot.
    '''
#    rfParams = {'n_estimators': 800,
#                'n_jobs': -1,
#                'max_leaf_nodes': 255, # important
#                'max_depth': 16,
#                'max_features': "sqrt",
#                'verbose': 1, 
#                'oob_score': True, 
#                'random_state': 2019}
#    X_train, X_test = pd.DataFrame(rep[10][1]), pd.DataFrame(rep[10][1])
##    X_train, X_test = pd.DataFrame(repBase[1]), pd.DataFrame(repBase[1])
#    X_train["target"] = np.where(groundTruth["label"] != -1, 0, 1)
#    rfScore, rfImportances, rfPred = rf_clf_training(X_train, X_test, rfParams=rfParams, numFolds=5, stratified=False)
#    plot_feature_importance(rfImportances, topK=20)
