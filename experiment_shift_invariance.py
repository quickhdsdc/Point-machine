#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:26:21 2019

@author: yinzhuo
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import seaborn as sns
import pandas as pd
import gc
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from UTILS import LoadSave
from DistanceMeasurements import dynamic_time_warping
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


def feature_kernel_pca(features=[], n_components=10):
    if len(features) == 0:
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True, axis=0)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))

    # Step 2: Get the features after the PCA processing
    pca = KernelPCA(n_components=n_components)
    featuresPCA = pca.fit_transform(features.values)
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
    Step 1: Load all DAE representations.
    '''
    PATH = ".//Data//rep_data//"
    ls = LoadSave(PATH)
    fileName = os.listdir(PATH)
    load_all_rep = False
    if load_all_rep:
        # Get the representations
        featureRep = feature_kernel_pca(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
        rep = []
        for name in fileName:
            rep.append(ls.load_data(PATH + name))
    
    ###############################################
    ###############################################
    '''
    Step 2: Shift invariance experiments.
    '''
    class_1_ind = groundTruth[groundTruth["label"] == 1]["ind"].values
    class_2_ind = groundTruth[groundTruth["label"] == 2]["ind"].values
    class_3_ind = groundTruth[groundTruth["label"] == 3]["ind"].values
    class_abnormal_ind = groundTruth[groundTruth["label"] == -1]["ind"].values
    
    rep_to_calculate = [rep[0], rep[1], rep[65], rep[16]]
    similarity_results = np.zeros((800, len(rep_to_calculate) + 2))
    
    '''
    Step 3: Draw samples.
    '''    
    # Draw samples from class 1
    #---------------------------------------------------
    np.random.seed(2019)
    samples_1 = np.random.choice(class_1_ind, 200, replace=True)
    
    np.random.seed(9102)
    samples_2 = np.random.choice(class_1_ind, 200, replace=True)
    for i, (ind_1, ind_2) in enumerate(zip(samples_1, samples_2)):
        ind_1, ind_2 = int(ind_1), int(ind_2)
        for j, item in enumerate(rep_to_calculate):
            similarity_results[i, j] = np.linalg.norm(item[ind_1, :] - item[ind_2, :])
        similarity_results[i, j+1] = dynamic_time_warping(ts[ind_1], ts[ind_2])
        similarity_results[i, j+2] = max(np.correlate(ts[ind_1], ts[ind_2], mode="full"))
        
        
    # Draw samples from class 2
    #---------------------------------------------------
    np.random.seed(520)
    samples_1 = np.random.choice(class_2_ind, 200, replace=True)
    
    np.random.seed(25)
    samples_2 = np.random.choice(class_2_ind, 200, replace=True)
    for i, (ind_1, ind_2) in enumerate(zip(samples_1, samples_2)):
        i = i + 200
        ind_1, ind_2 = int(ind_1), int(ind_2)
        for j, item in enumerate(rep_to_calculate):
            similarity_results[i, j] = np.linalg.norm(item[ind_1, :] - item[ind_2, :])
        similarity_results[i, j+1] = dynamic_time_warping(ts[ind_1], ts[ind_2])
        similarity_results[i, j+2] = max(np.correlate(ts[ind_1], ts[ind_2], mode="full"))
    
    
    # Draw samples from class abnormal
    #---------------------------------------------------
    np.random.seed(2912)
    samples_1 = np.random.choice(class_1_ind, 200, replace=True)
    
    np.random.seed(2504)
    samples_2 = np.random.choice(class_2_ind, 200, replace=True)
    for i, (ind_1, ind_2) in enumerate(zip(samples_1, samples_2)):
        i = i + 400
        ind_1, ind_2 = int(ind_1), int(ind_2)
        for j, item in enumerate(rep_to_calculate):
            similarity_results[i, j] = np.linalg.norm(item[ind_1, :] - item[ind_2, :])
        similarity_results[i, j+1] = dynamic_time_warping(ts[ind_1], ts[ind_2])
        similarity_results[i, j+2] = max(np.correlate(ts[ind_1], ts[ind_2], mode="full"))
    
    
    # Draw samples from class 3 with the abnormal data
    #---------------------------------------------------
    np.random.seed(666)
    samples_1 = np.random.choice(class_1_ind, 200, replace=True)
    
    np.random.seed(777)
    samples_2 = np.random.choice(class_3_ind, 200, replace=True)
    for i, (ind_1, ind_2) in enumerate(zip(samples_1, samples_2)):
        i = i + 600
        ind_1, ind_2 = int(ind_1), int(ind_2)
        for j, item in enumerate(rep_to_calculate):
            similarity_results[i, j] = np.linalg.norm(item[ind_1, :] - item[ind_2, :])
        similarity_results[i, j+1] = dynamic_time_warping(ts[ind_1], ts[ind_2])
        similarity_results[i, j+2] = max(np.correlate(ts[ind_1], ts[ind_2], mode="full"))
    
    X_sc = MinMaxScaler()
    similarity_results = X_sc.fit_transform(similarity_results)
    
    ###############################################
    ###############################################
    '''
    Step 4: Basic visualizing.
    '''
    plt.close("all")
    # Heat map for the shift invariance relationship
    feature = pd.DataFrame(similarity_results)
    corrList = [feature.loc[0:200].corr(), feature.loc[200:400].corr(),
                feature.loc[400:600].corr(), feature.loc[600:800].corr()]
    
    fig, axObj = plt.subplots(2, 2, figsize=(10, 8))
    for ind, (ax, featureCorr) in enumerate(zip(axObj.ravel(), corrList)):
        ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
                         cmap="Blues", fmt='.2f', annot=True,
                         annot_kws={'size':7.5,'weight':'bold'}, ax=ax)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        ax.tick_params(axis="x", labelsize=7)
        
        ax.set_xlabel("Variable X", fontsize=8)
        ax.set_ylabel("Variable Y", fontsize=8)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()

    
    # Linear relationship
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = sns.regplot(x=similarity_results[:, 1], y=similarity_results[:, -1], ax=ax, marker="o")
    ax = sns.regplot(x=similarity_results[:, 1], y=similarity_results[:, -2], ax=ax, marker="x")
    
    # Linear relationship
    fig, ax = plt.subplots(figsize=(10, 8))
    featureCorr = feature.corr()
    ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
                     cmap="Blues", fmt='.2f', annot=True,
                     annot_kws={'size':7.5,'weight':'bold'}, ax=ax)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.tick_params(axis="x", labelsize=7)
    
    ax.set_xlabel("Variable X", fontsize=8)
    ax.set_ylabel("Variable Y", fontsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    
    
#    # Sequence shift
#    #---------------------------------------------------
#    np.random.seed(9408)
#    sampled_seqs_1.append(np.random.choice(class_3_ind, 200, replace=True))
#    
#    np.random.seed(9410)
#    sampled_seqs_2.append(np.random.choice(class_abnormal_ind, 200, replace=True))
    