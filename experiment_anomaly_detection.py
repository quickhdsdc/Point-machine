#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:26:03 2019

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

from sklearn.decomposition import PCA
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
    Plot 1: DIfferent roc score under different lof nn parameters.
    '''
#    plt.close("all")
#    fig, axObj = plt.subplots(2, 2, figsize=(12, 9))
#    pts_index = [i for i in range(5, 250, 3)]
#    lw = 2
#    axObj[0][0].plot(featureRocRec[2][featureRocBestInd][1],
#                     featureRocRec[2][featureRocBestInd][2],
#                     label='FeatureBased(auc={:.3f})'.format(featureRocBest),
#                     lw=lw)    
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
#                         rocRec_ind[2][rocBestInd_ind][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocBest_ind),
#                         lw=lw)
#    axObj[0][0].legend(fontsize=8)
#    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    # 29(8)
#    index_0_1 = 30
#    pts = pts_index[index_0_1]
#    axObj[0][1].plot(featureRocRec[2][index_0_1][1],
#                     featureRocRec[2][index_0_1][2],
#                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_0_1]),
#                     lw=lw)
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[0][1].plot(rocRec_ind[2][index_0_1][1],
#                         rocRec_ind[2][index_0_1][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_0_1]),
#                         lw=lw)
#    axObj[0][1].legend(fontsize=8)
#    axObj[0][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#
#    # 59(18)
#    index_1_0 = 50
#    pts = pts_index[index_1_0]
#    axObj[1][0].plot(featureRocRec[2][index_1_0][1],
#                     featureRocRec[2][index_1_0][2],
#                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_1_0]),
#                     lw=lw)
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[1][0].plot(rocRec_ind[2][index_1_0][1],
#                         rocRec_ind[2][index_1_0][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_1_0]),
#                         lw=lw)
#    axObj[1][0].legend(fontsize=8)
#    axObj[1][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    # 80(25)
#    index_1_1 = 80
#    pts = pts_index[index_1_1]
#    axObj[1][1].plot(featureRocRec[2][index_1_1][1],
#                     featureRocRec[2][index_1_1][2],
#                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_1_1]),
#                     lw=lw)
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[1][1].plot(rocRec_ind[2][index_1_1][1],
#                         rocRec_ind[2][index_1_1][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_1_1]),
#                         lw=lw)
#    axObj[1][1].legend(fontsize=8)
#    axObj[1][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    for obj in axObj.ravel():
#        obj.grid(False)
#        obj.set_xlim(0, 1)
#        obj.set_ylim(0, )
#    plt.tight_layout()    
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//roc_curve_different_rep.png", dpi=500, bbox_inches="tight")