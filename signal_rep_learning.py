#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:00:29 2019

@author: yinzhuo
"""
# Disable or enable the GPU computation
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
from UTILS import generating_cutted_sequences
from UTILS import lgb_clf_training, plot_feature_importance

from StackedDenoisingAE import StackedDenoisingAE

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


def plot_clustering_results(data=[], c_label=[], methodStr="..//Plots//FeaturePcaBased//FeaturePcaBased_", plotNums=100):
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
        if plotNums > len(indexList):
            plotList = indexList
        else:
            for i in range(plotNums):
                plotList.append(random.choice(indexList))
        
        # Start plot
        f, ax = plt.subplots()
        for i in plotList:
            signal = data.iloc[i]["current"]
            plt.plot(signal, color="blue", linewidth=1.2, linestyle="-")
        plt.xlabel("time(s)")
        plt.ylabel("current(A)")
        plt.title("Cluster " + str(label) + ", samples " + str(len(indexList)))
        plt.grid(True)
        plt.savefig(methodStr + str(label) + ".png", dpi=500, bbox_inches="tight")
        plt.close("all")
    
    
def clustering_signals(features=[], n_clusters=2, method="kmeans"):
    if (len(features) == 0) or (n_clusters < 2):
        return None
  
    # Start clustering using the compressed data
    if method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, n_init=20, gamma=0.5,
                                 affinity="rbf", n_jobs=-1)
    else:    
        clf = KMeans(n_clusters=n_clusters, n_jobs=1)
    labels = clf.fit_predict(features)
    return labels


def data_encoding(data=[]):
    # Start padding the sequence
    numPts = 156
    newData = []
    for i in range(len(data)):
        tmp = data["current"].iloc[i]
        
        if len(tmp) > numPts:
            newData.append(tmp[:numPts])
        elif len(tmp) < numPts:
            paddingPtsNum = numPts - len(tmp)
            newData.append(np.pad(tmp, (0, paddingPtsNum), "edge"))
        else:
            newData.append(tmp)
    newData = np.array(newData)
    
    # Standardizing the data
    X_sc = MinMaxScaler()
    newData = X_sc.fit_transform(newData)
    
    trainPrecent = 0.9
    X_train, X_test = newData[:int(trainPrecent * len(newData)), :], newData[int(trainPrecent * len(newData)):, :]
    
    # Start the signal embedding
    n_hid = [60]
    sdae = StackedDenoisingAE(n_layers=len(n_hid), n_hid=n_hid, dropout=[0.01], nb_epoch=200, enc_act=["tanh"], dec_act=["linear"], batch_size=128)
    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(X_train, X_test, X_test, dir_out='..//Models//', write_model=False)
    gc.collect()
    return dense_train, dense_test


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


def plot_2d_gaussian_mixtrue_results(X, Y_, means, covariances, save):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
    f, ax = plt.subplots()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.5 * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], marker=".", s=10, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    plt.grid(True)
    plt.title("Components of GMM")
    if save == True:
        plt.savefig("..//Plots//gmm_components.png", dpi=500, bbox_inches="tight")

###############################################################################
###############################################################################
def dae_rep(trainData=None, vaildData=None, segRes=None, windowSize=None, stride=None,
            n_hid=[20], dropout=[0.02], enc_act=["relu"],
            path=".//Data//rep_data//", fileName="rep_0"):
    
    sdae = StackedDenoisingAE(n_layers=len(n_hid), n_hid=n_hid, dropout=[0.02],
                              nb_epoch=300, enc_act=["relu"], early_stop_rounds=50,
                              dec_act=["linear"], batch_size=256, bias=True)
    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(trainData, vaildData,
           vaildData, dir_out='.//Models//', write_model=False)

    # Combining the subsequences into a single sequence
    newData = np.concatenate([dense_train, dense_test], axis=0)
    newData = pd.DataFrame(newData)
    newData["flag"] = segRes["ind"]
    signalRep = newData.groupby(["flag"]).mean().values

    # Save the data
    ls = LoadSave(path + fileName + "_" + str(windowSize) + "_"
                  + str(stride) + ".pkl")
    ls.save_data(signalRep)
    return signalRep


###############################################################################
###############################################################################
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


def train_all_dae(X_train=None, X_valid=None, segRes=None,
                  windowSize=None, stride=None):
    cond_0 = True
    cond_1 = True
    cond_2 = True
    cond_3 = True
    
    if cond_0:
        '''
        Des: Increasing the neurons
        Code: ni
        '''
        n_hid, dropout, enc_act = [5], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ni_0")
        
        n_hid, dropout, enc_act = [10], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ni_1")
        
        n_hid, dropout, enc_act = [15], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ni_2")    

        n_hid, dropout, enc_act = [20], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ni_3")
    
    if cond_1:
        '''
        Des: Increasing the dropoutrate
        Code: di
        '''
        n_hid, dropout, enc_act = [15], [0.01], ["tanh"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_di_0")
        
        n_hid, dropout, enc_act = [15], [0.05], ["tanh"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_di_1")
        
        n_hid, dropout, enc_act = [15], [0.1], ["tanh"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_di_2")    

        n_hid, dropout, enc_act = [15], [0.2], ["tanh"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_di_3")
        
    if cond_2:
        '''
        Des: Different act fcn
        Code: ac
        '''
        n_hid, dropout, enc_act = [15], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ac_0")
        
        n_hid, dropout, enc_act = [15], [0.01], ["elu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ac_1")
        
        n_hid, dropout, enc_act = [15], [0.01], ["tanh"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ac_2")    

        n_hid, dropout, enc_act = [15], [0.01], ["sigmoid"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_ac_3")

    if cond_3:
        '''
        Des: Stacked layer
        Code: sl
        '''
        n_hid, dropout, enc_act = [15, 20], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_sl_0")
        
        n_hid, dropout, enc_act = [30, 20], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_sl_1")
        
        n_hid, dropout, enc_act = [40, 20], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_sl_2")    

        n_hid, dropout, enc_act = [100, 20], [0.01], ["relu"]
        newRep = dae_rep(X_train, X_valid, segRes, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_sl_3")
                
        
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


def load_all_dae_rep():
    pass

###############################################################################
###############################################################################
if __name__ == "__main__":
    currentData = load_data()
    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
    
    signal = {"current": ts}
    signal = pd.DataFrame(signal, columns=["current"])
    
    '''
    Step 1: Segmenting the time series and scale it.
    '''
    windowSize, stride = 60, 30
    segRes = sliding_window_signal_generating(data=ts, window=60, stride=30, portion=0.1)
    
    X_sc = MinMaxScaler()
    newData = np.array(segRes["segment"].values.tolist())
    length = [len(i) for i in newData]
    newData = X_sc.fit_transform(newData)
    ###############################################
    ###############################################    
    '''
    Step 2: Train all DAE.
    '''
#    trainPrecent = 0.85
#    X_train, X_test = newData[:int(trainPrecent * len(newData)), :], newData[int(trainPrecent * len(newData)):, :]   
#    train_all_dae(X_train, X_test, segRes, windowSize=windowSize, stride=stride)
    
    ##############################################
    ##############################################
    '''
    Step 3: Load all DAE representations, and perform the anamoly detection.
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
    featureRocBestInd, featureRocBest, featureRocRec = select_best_lof_value(data=featureRep,
                                                                             y_true=np.where(groundTruth["label"] != -1, 1, -1))
    rocBestInd, rocBest, rocRec = [], [], []
    for data_rep in rep:
        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep,
                                                    y_true=np.where(groundTruth["label"] != -1, 1, -1))
        rocBestInd.append(tmp_1)
        rocBest.append(tmp_2)
        rocRec.append(tmp_3)
    
    ###############################################
    ###############################################
    '''
    Step 3: Basic plottings.
    '''
    
    '''
    Different nn parameters(best, 30, 60, 80)
    '''
    plt.close("all")
    fig, axObj = plt.subplots(2, 2, figsize=(12, 9))
    pts_index = [i for i in range(5, 250, 3)]
    lw = 2
    axObj[0][0].plot(featureRocRec[2][featureRocBestInd][1],
                     featureRocRec[2][featureRocBestInd][2],
                     label='FeatureBased(auc={:.3f})'.format(featureRocBest),
                     lw=lw)    
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
                         rocRec_ind[2][rocBestInd_ind][2],
                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocBest_ind),
                         lw=lw)
    axObj[0][0].legend(fontsize=8)
    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    # 29(8)
    index_0_1 = 30
    pts = pts_index[index_0_1]
    axObj[0][1].plot(featureRocRec[2][index_0_1][1],
                     featureRocRec[2][index_0_1][2],
                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_0_1]),
                     lw=lw)
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][1].plot(rocRec_ind[2][index_0_1][1],
                         rocRec_ind[2][index_0_1][2],
                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_0_1]),
                         lw=lw)
    axObj[0][1].legend(fontsize=8)
    axObj[0][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # 59(18)
    index_1_0 = 50
    pts = pts_index[index_1_0]
    axObj[1][0].plot(featureRocRec[2][index_1_0][1],
                     featureRocRec[2][index_1_0][2],
                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_1_0]),
                     lw=lw)
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][0].plot(rocRec_ind[2][index_1_0][1],
                         rocRec_ind[2][index_1_0][2],
                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_1_0]),
                         lw=lw)
    axObj[1][0].legend(fontsize=8)
    axObj[1][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    # 80(25)
    index_1_1 = 80
    pts = pts_index[index_1_1]
    axObj[1][1].plot(featureRocRec[2][index_1_1][1],
                     featureRocRec[2][index_1_1][2],
                     label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][index_1_1]),
                     lw=lw)
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][1].plot(rocRec_ind[2][index_1_1][1],
                         rocRec_ind[2][index_1_1][2],
                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_1_1]),
                         lw=lw)
    axObj[1][1].legend(fontsize=8)
    axObj[1][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    axObj[1][1].set_xlabel('False Positive Rate(FPR)')
#    axObj[1][1].set_ylabel('True Positive Rate(TPR)')
    
    for obj in axObj.ravel():
        obj.grid(False)
        obj.set_xlim(0, 1)
        obj.set_ylim(0, )
    plt.tight_layout()
    
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
    