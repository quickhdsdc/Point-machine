#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:49:05 2019

@author: yinzhuo
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score
from keras.regularizers import l1, l2, l1_l2

from UTILS import LoadSave
from UTILS import generating_cutted_sequences
from UTILS import get_dae_rep, plot_roc_auc_curve

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


def sliding_window_signal_generating(data=[], window=60, stride=30, portion=0.4):
    if len(data) == 0:
        return None
    
    segRes = {"ind":[], "mean":[], "var":[], "segment":[], "length":[], "remainPrecent":[]}
    cuttedRes = generating_cutted_sequences(data=data, window=window, stride=stride, portion=0.4, verboseRound=500)
    
    for ind in range(len(cuttedRes)):
        tmp = cuttedRes[ind]
        for segInd in tmp.keys():
            segRes["ind"].append(ind)
            segRes["segment"].append(tmp[segInd][0])
            segRes["mean"].append(tmp[segInd][1])
            segRes["var"].append(tmp[segInd][2])
            segRes["length"].append(len(tmp[segInd][0]))
            segRes["remainPrecent"].append(tmp[segInd][3])
    segRes = pd.DataFrame(segRes)
    return segRes

###############################################################################
###############################################################################
if __name__ == "__main__":
#    currentData = load_data()
#    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
#    
#    signal = {"current": ts}
#    signal = pd.DataFrame(signal, columns=["current"])
#    
#    '''
#    Step 1: Segmenting the time series and scale it.
#    '''
#    windowSize, stride = 50, 50
#    segRes = sliding_window_signal_generating(data=ts, window=60, stride=30, portion=0.1)
#    
#    X_sc = MinMaxScaler()
#    newData = np.array(segRes["segment"].values.tolist())
#    length = [len(i) for i in newData]
#    newData = X_sc.fit_transform(newData)
#    ###############################################
#    ###############################################
#    trainPrecent = 0.85
#    X_train, X_test = newData[:int(trainPrecent * len(newData)), :], newData[int(trainPrecent * len(newData)):, :]
#    
#    # With activity regularization
#    rep_0 = get_dae_rep(X_train, X_test, segRes,
#                        windowSize=windowSize, stride=stride,
#                        n_hid=[20],
#                        weight_regularizer=[l2(0.0001)], activity_regularizer=[l1(0.0000001)])
#    
#    # Without activity regularization
#    rep_1 = get_dae_rep(X_train, X_test, segRes,
#                        windowSize=windowSize, stride=stride,
#                        n_hid=[20],
#                        weight_regularizer=[l2(0.0001)], activity_regularizer=[])
#
#    # Without activity regularization 
#    rep_2 = get_dae_rep(X_train, X_test, segRes,
#                        windowSize=windowSize, stride=stride,
#                        n_hid=[20],
#                        weight_regularizer=[l2(0.000001)], activity_regularizer=[])  
#
#    # Without activity regularization
#    rep_3 = get_dae_rep(X_train, X_test, segRes,
#                        windowSize=windowSize, stride=stride,
#                        n_hid=[20],
#                        weight_regularizer=[], activity_regularizer=[])
    ###############################################
    ###############################################
    '''
    LOF for anamoly detection.
    '''
    # rep_0
    clf = LocalOutlierFactor(n_neighbors=300, contamination=0.08, n_jobs=-1)
    noiseLabels = clf.fit_predict(rep_0)
    noiseLevel = clf.negative_outlier_factor_.reshape((len(groundTruth), 1))
    
    y_true = np.where(groundTruth["label"] != -1, 1, -1).reshape((len(groundTruth), 1))
    plot_roc_auc_curve(y_true, noiseLevel)
    
    # rep_1
    clf = LocalOutlierFactor(n_neighbors=300, contamination=0.08, n_jobs=-1)
    noiseLabels = clf.fit_predict(rep_1)
    noiseLevel = clf.negative_outlier_factor_.reshape((len(groundTruth), 1))
    plot_roc_auc_curve(y_true, noiseLevel)

    # rep_2
    clf = LocalOutlierFactor(n_neighbors=300, contamination=0.08, n_jobs=-1)
    noiseLabels = clf.fit_predict(rep_2)
    noiseLevel = clf.negative_outlier_factor_.reshape((len(groundTruth), 1))
    plot_roc_auc_curve(y_true, noiseLevel)

    # rep_3
    clf = LocalOutlierFactor(n_neighbors=300, contamination=0.08, n_jobs=-1)
    noiseLabels = clf.fit_predict(rep_3)
    noiseLevel = clf.negative_outlier_factor_.reshape((len(groundTruth), 1))
    plot_roc_auc_curve(y_true, noiseLevel)