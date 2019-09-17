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
import numpy as np
import seaborn as sns
import pandas as pd
import gc
import random
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score

from UTILS import LoadSave
from UTILS import generating_cutted_sequences, generating_equal_sequences
from UTILS import timefn
from keras.regularizers import l1, l2, l1_l2

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


# Warp for the generating_cutted_sequences.py
@timefn
def signal_dataframe_generating(cuttedRes=[], window=60,
                                stride=30, portion=0.4,
                                bins=30):
    if len(cuttedRes) == 0:
        return None
    
    segRes = {"ind":[], "mean":[], "var":[], "segment":[], "length":[], "remainPrecent":[]}
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
    
    # Get the norm factor
    segRes["bins"] = pd.cut(segRes["var"].values, bins=np.linspace(segRes["var"].min(), segRes["var"].max(), num=25), include_lowest=True, labels=False)
    featureCount = segRes.groupby(["bins"])["length"].count().reset_index().rename({"length":"bin_freq"}, axis=1)
    segRes = pd.merge(segRes, featureCount, how="left", on="bins")
    segRes["bin_freq"] = segRes["bin_freq"] / len(segRes)
    
    # Defining the sample weigths
    #-------------------------------------------------------
#    segRes["weights"] = -np.log(segRes["bin_freq"].values)
    segRes["weights"] = 1 / (1 + np.exp(0.1 * segRes["bin_freq"].values))
    #-------------------------------------------------------
    
    weightsNorm = segRes.groupby(["ind"])["weights"].sum().reset_index().rename({"weights":"norm_factor"}, axis=1)
    segRes = pd.merge(segRes, weightsNorm, how="left", on="ind")
    segRes["weights_norm"] = segRes["weights"] / segRes["norm_factor"]
    
    if window:
        print("@Window {}, stride {}, average remain precent {:.3f}.".format(window, stride, segRes["remainPrecent"].mean()))
    return segRes

@timefn
def get_dae_rep(trainData=None, vaildData=None, segDataFrame=None,
                windowSize=None, stride=None,
                weight_regularizer=[None], activity_regularizer=[None],
                n_hid=[20], dropout=[0.02], nb_epoch=500, batch_size=512,
                early_stop_rounds=50, 
                enc_act=["relu"], dec_act=["linear"],
                path=".//Data//rep_data//", fileName="rep_0", save_rep=False):
    
    # Check the parameters
    if windowSize == None or stride == None:
        return
    
    # Train the dae rep
    sdae = StackedDenoisingAE(n_layers=len(n_hid), n_hid=n_hid, dropout=dropout,
                              weight_regularizer=weight_regularizer,
                              activity_regularizer=activity_regularizer,
                              nb_epoch=nb_epoch, enc_act=enc_act,
                              early_stop_rounds=early_stop_rounds, dec_act=dec_act,
                              batch_size=batch_size, bias=True)
    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(trainData, vaildData,
           vaildData, dir_out='.//Models//', write_model=False)

    # Combining the subsequences into a single sequence(weighted or non-weighted)
    newData = np.concatenate([dense_train, dense_test], axis=0)
    newData = pd.DataFrame(newData)
    
    # Weighted new data
    newData_weight = newData.multiply(segDataFrame["weights_norm"].values, axis=0).copy()
    newData_weight["flag"] = segDataFrame["ind"]
    signalRep_weight = newData_weight.groupby(["flag"]).mean().values
    
    # Unweighted 
    newData["flag"] = segDataFrame["ind"]
    signalRep = newData.groupby(["flag"]).mean().values

    # Save the data
    if save_rep:
        ls = LoadSave(path + fileName + "_" + str(windowSize) + "_"
                      + str(stride) + ".pkl")
        ls.save_data(signalRep)
        
        ls = LoadSave(path + fileName + "_" + str(windowSize) + "_"
              + str(stride) + "_weight" + ".pkl")
        ls.save_data(signalRep_weight)
    return signalRep


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
###############################################################################
###############################################################################
# Number of neurons, and the dropout rate
def train_baseline_dae(X_train=None, X_valid=None, segDataFrame=None):
    '''
    Baseline 0
    '''
    newRep = get_dae_rep(X_train, X_valid, segDataFrame,
                         windowSize=0, stride=0,
                         weight_regularizer=[l2(0.0001)], activity_regularizer=[l1(0.000001)],
                         n_hid=[20], dropout=[0.02], nb_epoch=1500, batch_size=512,
                         early_stop_rounds=50, 
                         enc_act=["relu"], fileName="rep_base_0", save_rep=True)
    
    '''
    Baseline 1
    '''
    newRep = get_dae_rep(X_train, X_valid, segDataFrame,
                         windowSize=0, stride=0,
                         weight_regularizer=[None], activity_regularizer=[l1(0.00001)],
                         n_hid=[60, 30], dropout=[0.05], nb_epoch=1500, batch_size=512,
                         early_stop_rounds=50,
                         enc_act=["relu"], fileName="rep_base_1", save_rep=True)
    
    
def train_rnn_dae():
    pass


def train_rnn_simulation(X_train=None, X_valid=None, segDataFrame=None,
                         windowSize=None, stride=None):
    '''
    RNN 0
    '''
    n_hid, dropout, enc_act = [20], [0.01], ["elu"]
    newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_rnn_0", save_rep=True)
    '''
    RNN 1
    '''
    n_hid, dropout, enc_act = [25], [0.1], ["elu"]
    newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                         n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                         fileName="rep_rnn_1", save_rep=True)


def train_mlp_dae(X_train=None, X_valid=None, segDataFrame=None,
                  windowSize=None, stride=None):
    '''
    @Description:
    ----------
        --type_0: Increasing the neurons, constant dropout rate.(Code: in, DropoutRate=0.01)
        --type_1: Increasing the dropout rate, constant neurons.(Code: id, NN=20)
        --type_2: Stacked the layer(Code: sl)
    '''
    
    type_0 = True
    type_1 = True
    type_2 = False
    
    # Train all MLP Autoencoders.
    #--------------------------------------------------------
    if type_0:
        '''
        Description: Increasing the neurons, constant dropout rate.
        Code: in
        '''
        n_hid, dropout, enc_act = [5], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_0", save_rep=True)        
        
        n_hid, dropout, enc_act = [10], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_1", save_rep=True)
        
        n_hid, dropout, enc_act = [15], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_2", save_rep=True)

        n_hid, dropout, enc_act = [20], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_3", save_rep=True)
        
        n_hid, dropout, enc_act = [25], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_4", save_rep=True)

        n_hid, dropout, enc_act = [30], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_5", save_rep=True)

        n_hid, dropout, enc_act = [35], [0.01], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_in_6", save_rep=True)


    if type_1:
        '''
        Description: Increasing the dropout rate, constant neurons.
        Code: id
        '''
        n_hid, dropout, enc_act = [20], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_0", save_rep=True)        
        
        n_hid, dropout, enc_act = [20], [0.07], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_1", save_rep=True)
        
        n_hid, dropout, enc_act = [20], [0.1], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_2", save_rep=True)

        n_hid, dropout, enc_act = [20], [0.13], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_3", save_rep=True)
        
        n_hid, dropout, enc_act = [20], [0.16], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_4", save_rep=True)

        n_hid, dropout, enc_act = [20], [0.20], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_5", save_rep=True)

        n_hid, dropout, enc_act = [20], [0.25], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_id_6", save_rep=True)


    if type_2:
        '''
        Description: Increasing the dropout rate, constant neurons.
        Code: id
        '''
        n_hid, dropout, enc_act = [100, 100], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_0", save_rep=True)        
        
        n_hid, dropout, enc_act = [100, 50], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_1", save_rep=True)
        
        n_hid, dropout, enc_act = [100, 20], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_2", save_rep=True)

        n_hid, dropout, enc_act = [100, 50, 20], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_3", save_rep=True)
        
        n_hid, dropout, enc_act = [20, 50, 100], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_4", save_rep=True)

        n_hid, dropout, enc_act = [20, 20], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_5", save_rep=True)

        n_hid, dropout, enc_act = [20, 100], [0.05], ["tanh"]
        newRep = get_dae_rep(X_train, X_valid, segDataFrame, windowSize=windowSize, stride=stride,
                             n_hid=n_hid, dropout=dropout, enc_act=enc_act,
                             fileName="rep_sl_6", save_rep=True)
    return

def train_cross_dae():
    pass


###############################################################################
###############################################################################
if __name__ == "__main__":
    ###############################################
    ###############################################
    '''
    Step 0: Set some hyper parameters.
    
    '''
    plotShow, plotSave = False, True
    currentData = load_data()
    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
    
    signal = {"current": ts}
    signal = pd.DataFrame(signal, columns=["current"])
    
    ###############################################
    ###############################################
    '''
    Step 1: Segmenting the time series and scale it.
    '''
    splitData = True
    if splitData:
        segDataFrameList, windowParamsList = [], []
        bins = 10
        # Segment 1
#        windowSize, stride = 40, 10
#        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
#                                                stride=stride, portion=0.4,
#                                                verboseRound=500)
#        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
#                                                stride=stride, bins=bins)
#        segDataFrameList.append(cuttedRes)
#        windowParamsList.append([windowSize, stride])
#        
#        # Segment 2
#        windowSize, stride = 40, 20
#        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
#                                                stride=stride, portion=0.4,
#                                                verboseRound=500)
#        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
#                                                stride=stride, bins=bins)
#        segDataFrameList.append(cuttedRes)
#        windowParamsList.append([windowSize, stride])
#        
#        # Segment 3
#        windowSize, stride = 40, 30
#        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
#                                                stride=stride, portion=0.4,
#                                                verboseRound=500)
#        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
#                                                stride=stride, bins=bins)
#        segDataFrameList.append(cuttedRes)
#        windowParamsList.append([windowSize, stride])
#        
        # Segment 4
        windowSize, stride = 50, 10
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 5
        windowSize, stride = 50, 20
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 6
        windowSize, stride = 50, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 7
        windowSize, stride = 60, 10
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 8
        windowSize, stride = 60, 20
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 9
        windowSize, stride = 60, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, bins=bins)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 10, using as the baseline 
        cuttedRes = generating_equal_sequences(data=ts, mode="mean_length")
        cuttedRes = signal_dataframe_generating(cuttedRes, window=None)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([0, 0])
        
        del cuttedRes
        gc.collect()
        
        # Normalizing the data
        segTsList = []
        for df in segDataFrameList:
            X_sc = MinMaxScaler()
            dataTmp = np.array(df["segment"].values.tolist())
            dataTmp = X_sc.fit_transform(dataTmp)
            segTsList.append(dataTmp)
    ###############################################
    ###############################################
    '''
    Step 2: Train all representations.
    '''
    trainPrecent = 0.85
    # Training the baseline model
    #----------------------------------------------------
    X_train, X_test = segTsList[-1][:int(trainPrecent * len(segTsList[-1])), :], segTsList[-1][int(trainPrecent * len(segTsList[-1])):, :]
    train_baseline_dae(X_train, X_test, segDataFrameList[-1])
    
    # Training the original repsentation data
    #----------------------------------------------------
#    for ind, (segTs, segDataFrame, windowParam) in enumerate(zip(segTsList[:-1], segDataFrameList[:-1], windowParamsList[:-1])):
#        X_train, X_test = segTs[:int(trainPrecent * len(segTs)), :], segTs[int(trainPrecent * len(segTs)):, :]
#        train_mlp_dae(X_train, X_test, segDataFrame,
#                      windowSize=windowParam[0], stride=windowParam[1])
#        
#    train_rnn_dae()
    ###############################################
    ###############################################
    '''
    Step 3: Train all DAE.
    '''
#    test_ind = 0
#    newData, flag, weights = segTsList[test_ind], segDataFrameList[test_ind]["ind"].values, segDataFrameList[test_ind]["weights_norm"].values
#    trainPrecent = 0.85
#    X_train, X_test = newData[:int(trainPrecent * len(newData)), :], newData[int(trainPrecent * len(newData)):, :]
#    sdae = StackedDenoisingAE(n_layers=1, n_hid=[25], dropout=[0.02],
#                              weight_regularizer=[None],
#                              activity_regularizer=[None],
#                              nb_epoch=700, early_stop_rounds=50,
#                              enc_act=["elu"], dec_act=["linear"],
#                              batch_size=1024, bias=True)
#    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(X_train, X_test,
#           X_test, dir_out='.//Models//', write_model=False)
#
#    # Combining the subsequences into a single sequence
#    newData = np.concatenate([dense_train, dense_test], axis=0)
#    newData = pd.DataFrame(newData)
#    newData["flag"] = flag
#    signalRep_mean = newData.groupby(["flag"]).mean().values
#    
#    newData = newData.multiply(weights, axis=0)
#    newData["flag"] = flag
#    signalRep_weight = newData.groupby(["flag"]).mean().values
    
    ##############################################
    ##############################################
    '''
    Step 3: Basic visualizing for the code evaluation.
    '''
    # Normalizing the data first
#    X_sc = StandardScaler()
#    rep_mean = X_sc.fit_transform(signalRep_mean)
#    rep_weight = X_sc.fit_transform(signalRep_weight)
#    
#    # -----------------------------------------------------------
#    # Visualizing the 2-d distribution in feature space
#    # kpca ===>> 2 components
#    components = 2
#    pca = KernelPCA(n_components=components, kernel='rbf')
#    rep_mean = pca.fit_transform(rep_mean)
#    rep_weight = pca.fit_transform(rep_weight)
#    
#    plt.close("all")
#    f, axObj = plt.subplots(1, 2, figsize=(16, 6))
#    uniqueLabels = [1, 2, 3, -1]
#    for ind, label in enumerate(uniqueLabels):
#        sampleIndex = np.arange(0, len(groundTruth))
#        labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
#        
#        coords_mean = rep_mean[labeledSampleIndex, :]
#        coords_weight = rep_weight[labeledSampleIndex, :]
#        if label != -1:
#            axObj[0].scatter(coords_mean[:, 0], coords_mean[:, 1], s=10,
#                 color=colors[ind], marker=".", label="Class " + str(label))
#            axObj[1].scatter(coords_weight[:, 0], coords_weight[:, 1], s=10,
#                 color=colors[ind], marker=".", label="Class " + str(label))
#        else:
#            axObj[0].scatter(coords_mean[:, 0], coords_mean[:, 1], s=10, color='r', marker="x", label="Class Abnormal")
#            axObj[1].scatter(coords_weight[:, 0], coords_weight[:, 1], s=10, color='r', marker="x", label="Class Abnormal")    
#        axObj[0].tick_params(axis="both", labelsize=10)
#        axObj[1].tick_params(axis="both", labelsize=10)
#        
#        axObj[0].legend(fontsize=10)
#        axObj[1].legend(fontsize=10)
#        
#    if plotSave:
#        plt.savefig(".//Plots//1_REP_kpca_rep_distribution.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")        
#    # -----------------------------------------------------------
#    # LOF for the anomaly detection
#    rocBestInd, rocBest, rocRec = [], [], []
#    for data_rep in [signalRep_mean, signalRep_weight]:
#        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep, y_true=np.where(groundTruth["label"] != -1, 1, -1))
#        rocBestInd.append(tmp_1)
#        rocBest.append(tmp_2)
#        rocRec.append(tmp_3)
#    
#    fig, axObj = plt.subplots(2, 2, figsize=(12, 9))
#    pts_index = [i for i in range(100, 200, 3)]
#    lw = 2
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
#                         rocRec_ind[2][rocBestInd_ind][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocBest_ind),
#                         lw=lw)
#    axObj[0][0].legend(fontsize=8)
#    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    # 29(8)
#    index_0_1 = 5
#    pts = pts_index[index_0_1]
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[0][1].plot(rocRec_ind[2][index_0_1][1],
#                         rocRec_ind[2][index_0_1][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_0_1]),
#                         lw=lw)
#    axObj[0][1].legend(fontsize=8)
#    axObj[0][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#
#    # 59(18)
#    index_1_0 = 20
#    pts = pts_index[index_1_0]
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[1][0].plot(rocRec_ind[2][index_1_0][1],
#                         rocRec_ind[2][index_1_0][2],
#                         label='Rep_' + str(ind) + '(auc={:.3f})'.format(rocRec_ind[1][index_1_0]),
#                         lw=lw)
#    axObj[1][0].legend(fontsize=8)
#    axObj[1][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    # 80(25)
#    index_1_1 = 30
#    pts = pts_index[index_1_1]
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
#    if plotSave:
#        plt.savefig(".//Plots//1_REP_roc_auc_plot.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#        
#    # -----------------------------------------------------------
#    # Heatmap of the rep
#    fig, axObj = plt.subplots(1, 2, figsize=(10, 4))
#    for ind, (ax, rep) in enumerate(zip(axObj, [signalRep_mean, signalRep_weight])):
#        tmp = pd.DataFrame(rep)
#        featureCorr = tmp.corr()
#        
#        ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
#                         cmap="Blues", fmt='.2f', annot=True,
#                         annot_kws={'size':4.5,'weight':'bold'}, ax=ax)
#        ax.tick_params(axis="y", labelsize=7, rotation=0)
#        ax.tick_params(axis="x", labelsize=7)
#        
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=7)    
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//1_REP_rep_corr.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#    