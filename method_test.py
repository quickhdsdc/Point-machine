#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:34:37 2019

@author: yinzhuo
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
import gc
from datetime import datetime
from tqdm import tqdm
import random
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score

from UTILS import LoadSave
from UTILS import timefn

import keras
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Dropout, Lambda
from keras.models import Model, Sequential, model_from_json
from keras.layers.recurrent import GRUCell 
from keras.preprocessing.sequence import  pad_sequences

from UTILS import generating_cutted_sequences, generating_equal_sequences
from StackedDenoisingAE import StackedDenoisingAE

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["C" + str(i) for i in range(0, 9+1)]
markers = ["s", "^", "o", "d", "*"]
#mpl.rcParams["font.family"] = "Times New Roman"
###############################################################################
###############################################################################

# Warp for the generating_cutted_sequences.py
@timefn
def signal_dataframe_generating(cuttedRes=[], window=60,
                                stride=30, portion=0.4,
                                binsList=[5, 10, 15]):
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
    
    # Generating the bin intervels
    binNameList = []
    for intervel in binsList:
        segRes["bins_" + str(intervel)] = pd.cut(segRes["var"].values,
               bins=np.linspace(segRes["var"].min(),
                                segRes["var"].max(),
                                num=intervel),
               include_lowest=True, labels=False)
        binNameList.append("bins_" + str(intervel))
    
    # Generating the bin_freq for each intervel
    for ind, (binName, intervel) in enumerate(zip(binNameList, binsList)):
        featureCount = segRes.groupby([binName])["length"].count().reset_index().rename({"length":"bin_freq_" + str(intervel)}, axis=1)
        segRes = pd.merge(segRes, featureCount, how="left", on=binName)
        segRes["bin_freq_" + str(intervel)] = segRes["bin_freq_" + str(intervel)] / len(segRes)
        segRes.drop(binName, axis=1, inplace=True)
    
    if window:
        print("@Window {}, stride {}, average remain precent {:.3f}.".format(window, stride, segRes["remainPrecent"].mean()))
    return segRes


def select_best_lof_value(data=None, y_true=None,
                          nn_range=[i for i in range(60, 300, 5)]):
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


def load_data():
    fileName = ["current_ts_merged_3300.pkl"]
    ls = LoadSave()
    currentData = ls.load_data(".//Data//" + fileName[0])
    return currentData


def signal_2_vec(trainData=None, validData=None, segDataFrame=None,
                 windowSize=None, stride=None,
                 n_hid=50, nb_epoch=100, batch_size=512,
                 early_stop_rounds=200,
                 path=".//Data//rep_data//", fileName="rep_0", save_rep=True):
    
    '''
    Step 0: Preparing the data.
    '''    
    K.clear_session()
    train_left, train_center, train_right = np.array([item[0] for item in trainData]), np.array([item[1] for item in trainData]), np.array([item[2] for item in trainData])
    valid_left, valid_center, valid_right = np.array([item[0] for item in validData]), np.array([item[1] for item in validData]), np.array([item[2] for item in validData])
    
    '''
    Step 1: Build the Input and output.
    '''
    input_left = Input(shape=(windowSize, ))
    input_center = Input(shape=(windowSize, ))
    input_right = Input(shape=(windowSize, ))
    input_all = [input_left, input_center, input_right]
    
    input_to_dense = Lambda(lambda x: K.sum(x, axis=0))(input_all)
    dense_hidden = Dense(n_hid, activation='elu')(input_to_dense)
    dense_output = Dense(windowSize, activation='linear')(dense_hidden)
    
    '''
    Step 2: Build the Model.
    '''
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=early_stop_rounds,
                                   verbose=0)
    model = Model([input_left, input_center, input_right], dense_output)
    model.compile(optimizer='adam', loss="mae")
    model.summary()
    
    '''
    Step 3: Fit the Model.
    '''    
    model.fit([train_left, train_center, train_right], train_center,
              validation_data=[[valid_left, valid_center, valid_right], valid_center],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks=[early_stopping])
    
    encoder = Model([input_left, input_center, input_right], dense_hidden)
    dense_train = encoder.predict([train_left, train_center, train_right])
    dense_valid = encoder.predict([valid_left, valid_center, valid_right])
    
    repTs = np.concatenate([dense_train, dense_valid], axis=0)
    repTs = pd.DataFrame(repTs)
    repTs.columns = ["rep_" + str(i) for i in range(repTs.shape[1])]
    
    segDataFrame = pd.concat([segDataFrame, repTs], axis=1)
    
    '''
    Step 4: Save the representation data.
    '''
    if save_rep:
        ls = LoadSave(path + fileName + "_" + str(windowSize) + "_"
                      + str(stride) + ".pkl")
        ls.save_data(segDataFrame)

    return segDataFrame
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
    y_true=np.where(groundTruth["label"] != -1, 1, -1)
    ###############################################
    ###############################################
    '''
    Step 1: Segmenting the time series and scale it.
    '''
    splitData = False
    if splitData:
        segDataFrameList, windowParamsList = [], []
        binsList = [i for i in range(5, 50+5, 5)]
        
        # Segment 0
        windowSize, stride = 30, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 1
        windowSize, stride = 40, 20
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 2
        windowSize, stride = 40, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 3
        windowSize, stride = 50, 10
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 4
        windowSize, stride = 50, 20
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 5
        windowSize, stride = 50, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 6
        windowSize, stride = 60, 10
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 7
        windowSize, stride = 60, 20
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 8
        windowSize, stride = 60, 30
        cuttedRes = generating_cutted_sequences(data=ts, window=windowSize,
                                                stride=stride, portion=0.4,
                                                verboseRound=500)
        cuttedRes = signal_dataframe_generating(cuttedRes, window=windowSize,
                                                stride=stride, binsList=binsList)
        segDataFrameList.append(cuttedRes)
        windowParamsList.append([windowSize, stride])
        
        # Segment 9, using as the baseline 
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
    test_ind = 3
    windowSize, stride = 50, 10
    segDf = segDataFrameList[test_ind]
    
    segTsCenterIndex, segTs = [], []
    pointer, value = 0, 0
    while(pointer < len(segDf)):
        if pointer == (len(segDf) - 1) or (segDf["ind"].loc[pointer] != segDf["ind"].loc[pointer + 1]):
            tmp = pad_sequences([segDf["segment"].loc[pointer][-stride:]], 
                                dtype='float64', padding='post', value=value,
                                maxlen=windowSize)            
            segTs.append([segDf["segment"].loc[pointer-1],
                             segDf["segment"].loc[pointer],
                             tmp[0].tolist()])
            segTsCenterIndex.append(pointer)
            pointer += 1
            continue
        
        # Front of df
        if (pointer == 0) or (segDf["ind"].loc[pointer] != segDf["ind"].loc[pointer - 1]):
            tmp = pad_sequences([segDf["segment"].loc[pointer][:stride]],
                                dtype='float64', padding='pre', value=value
                                , maxlen=windowSize)
            segTs.append([tmp[0].tolist(),
                             segDf["segment"].loc[pointer],
                             segDf["segment"].loc[pointer+1]])
            segTsCenterIndex.append(pointer)
            pointer += 1
        else:
            segTs.append([segDf["segment"].loc[pointer-1], 
                             segDf["segment"].loc[pointer],
                             segDf["segment"].loc[pointer+1]])
            segTsCenterIndex.append(pointer)
            pointer += 1
    
    trainPrecent = 0.8
    X_train, X_test = segTs[:int(trainPrecent * len(segTs))], segTs[int(trainPrecent * len(segTs)):]
    
    segDataFrame = signal_2_vec(trainData=X_train, validData=X_test,
                                segDataFrame=segDf, n_hid=60,
                                windowSize=windowSize, stride=stride,
                                batch_size=len(X_train), nb_epoch=1500)
    ##############################################
    ##############################################
    # BEST ROC: test_ind=-2, norm_factor= 0.1, bin_freq_50
    # Picturing Best: test_ind=1, norm_factor= 0.05, bin_freq_5
    norm_factor, binName = 0.05, "bin_freq_" + str(10)
    repTs = segDataFrame[[name for name in segDataFrame.columns if "rep" in name]]
    segRes = segDataFrame[[name for name in segDataFrame.columns if "rep" not in name]]
    
    # Defining the sample weigths
    #-------------------------------------------------------    
    segRes["weights"] = 2 / (1 + np.exp(norm_factor * segRes[binName].values))
    #-------------------------------------------------------
    
    weightsNorm = segRes.groupby(["ind"])["weights"].sum().reset_index().rename({"weights":"norm_factor"}, axis=1)
    segRes = pd.merge(segRes, weightsNorm, how="left", on="ind")
    segRes["weights_norm"] = segRes["weights"] / segRes["norm_factor"]    
    
    # signalRep_weight
    signalRep_weight = repTs.multiply(segRes["weights_norm"].values, axis=0)
    signalRep_weight["flag"] = segRes["ind"]
    signalRep_weight = signalRep_weight.groupby(["flag"]).mean().values
    
    # signalRep_mean
    repTs["ind"] = segRes["ind"].values
    signalRep_mean = repTs.groupby(["ind"]).mean().values    
    ##############################################
    ##############################################
    '''
    Step 3: Basic visualizing for the code evaluation.
    '''
    # Normalizing the data first
    X_sc = StandardScaler()
    rep_mean = X_sc.fit_transform(signalRep_mean)
    rep_weight = X_sc.fit_transform(signalRep_weight)
    
    # -----------------------------------------------------------
    # Visualizing the 2-d distribution in feature space
    # kpca ===>> 2 components
    components = 2
    pca = KernelPCA(n_components=components, kernel='rbf')
    rep_mean = pca.fit_transform(rep_mean)
    rep_weight = pca.fit_transform(rep_weight)
    
    plt.close("all")
    f, axObj = plt.subplots(1, 2, figsize=(16, 6))
    uniqueLabels = [1, 2, 3, -1]
    for ind, label in enumerate(uniqueLabels):
        sampleIndex = np.arange(0, len(groundTruth))
        labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
        
        coords_mean = rep_mean[labeledSampleIndex, :]
        coords_weight = rep_weight[labeledSampleIndex, :]
        if label != -1:
            axObj[0].scatter(coords_mean[:, 0], coords_mean[:, 1], s=8, alpha=0.4,
                 color=colors[ind], marker=markers[ind], label="Class " + str(label))
            axObj[1].scatter(coords_weight[:, 0], coords_weight[:, 1], s=8, alpha=0.4,
                 color=colors[ind], marker=markers[ind], label="Class " + str(label))
        else:
            axObj[0].scatter(coords_mean[:, 0], coords_mean[:, 1], s=8,
                 alpha=0.4, color='r', marker="x", label="Class Abnormal")
            axObj[1].scatter(coords_weight[:, 0], coords_weight[:, 1], s=8,
                 alpha=0.4, color='r', marker="x", label="Class Abnormal")    
        axObj[0].tick_params(axis="both", labelsize=8)
        axObj[1].tick_params(axis="both", labelsize=8)
        
        axObj[0].legend(fontsize=8)
        axObj[1].legend(fontsize=8)
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//1_REP_kpca_rep_distribution_" + str(norm_factor)
        + "_" + binName + ".png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")        
    # -----------------------------------------------------------
    # LOF for the anomaly detection
    rocBestInd, rocBest, rocRec = [], [], []
    for data_rep in [signalRep_mean, signalRep_weight]:
        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep, y_true=y_true,
                                                    nn_range=[i for i in range(20, 200, 10)])
        rocBestInd.append(tmp_1)
        rocBest.append(tmp_2)
        rocRec.append(tmp_3)
    
    fig, axObj = plt.subplots(2, 2, figsize=(12, 9))
    pts_index = [i for i in range(20, 200, 10)]
    lw = 2
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
                         rocRec_ind[2][rocBestInd_ind][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocBest_ind),
                         lw=lw)
    axObj[0][0].legend(fontsize=7)
    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    # 29(8)
    index_0_1 = 2
    pts = pts_index[index_0_1]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][1].plot(rocRec_ind[2][index_0_1][1],
                         rocRec_ind[2][index_0_1][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_0_1]),
                         lw=lw)
    axObj[0][1].legend(fontsize=7)
    axObj[0][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    # 59(18)
    index_1_0 = 9
    pts = pts_index[index_1_0]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][0].plot(rocRec_ind[2][index_1_0][1],
                         rocRec_ind[2][index_1_0][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_1_0]),
                         lw=lw)
    axObj[1][0].legend(fontsize=7)
    axObj[1][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    # 80(25)
    index_1_1 = 17
    pts = pts_index[index_1_1]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][1].plot(rocRec_ind[2][index_1_1][1],
                         rocRec_ind[2][index_1_1][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_1_1]),
                         lw=lw)
    axObj[1][1].legend(fontsize=7)
    axObj[1][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    for obj in axObj.ravel():
        obj.grid(False)
        obj.set_xlim(0, 1)
        obj.set_ylim(0, )
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//1_REP_roc_auc_plot_" + str(norm_factor)
        + "_" + binName + ".png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")