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

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

from keras.regularizers import l1, l2, l1_l2
from UTILS import LoadSave, weighting_rep
from StackedDenoisingAE import StackedDenoisingAE

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["C" + str(i) for i in range(0, 9+1)]
markers = ["s", "^", "o", "d", "*", "<", ">", "h", "p"]
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
    pca = KernelPCA(n_components=n_components)
    featuresPCA = pca.fit_transform(features.values)
    
    return featuresPCA

###############################################################################
###############################################################################
if __name__ == "__main__":
#    currentData = load_data()
#    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
#    y_true = np.where(groundTruth["label"] != -1, 1, -1)
#    
#    signal = {"current": ts}
#    signal = pd.DataFrame(signal, columns=["current"])
#    plotSave, plotShow = True, False
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
    load_all = False
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
        
#    ###############################################
#    ###############################################
#    # Anamoly detection: select the best score
#    # Step ==> 10, 20, 30, neurons(20), windowSize(40)
#    #----------------------------------------------------
    repList = [rep[18][0],  rep[19][0], repRNN[0], repRNN[1],
               repBase[0], repBase[1], rep[18][1],  rep[19][1]]
    repName = ["Average-20-20", "Average-30-30", "GRU-40", "GRU-70",
               "DAE-80", "SDAE-100-50", "Weighted-20-20", "Weighted-30-30"]

    # Anamoly detection(LOF): select the best score
    featureRocBestInd, featureRocBest, featureRocRec = select_best_lof_value(data=featureRep,
                                                                             y_true=y_true,
                                                                             nn_range=[i for i in range(60, 200, 10)])
    
    rocBestInd, rocBest, rocRec = [], [], []
    for data_rep in repList:
        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep,
                                                    y_true=y_true,
                                                    nn_range=[i for i in range(60, 200, 10)])
        rocBestInd.append(tmp_1)
        rocBest.append(tmp_2)
        rocRec.append(tmp_3)
    ###############################################
    ###############################################
    '''
    Plot 0: Weigthed function curve shape.
    '''
#    plt.close("all")
#    fig, ax = plt.subplots(figsize=(6, 4))
#    x_range = np.linspace(0, 1, 1000)
#    norm_factor = [5, 1, 0.5, 0.1, 0.05, 0.01]
#    for ind, factor in enumerate(norm_factor):
#        res = 2 / (1 + np.exp(factor * x_range))
#        ax.plot(x_range, res, color=colors[ind], lw=2, label="Norm factor: {}".format(factor))
#        ax.set_xlim(0, 1)
#        ax.legend(fontsize=7, loc='lower left')
#        ax.tick_params(axis="both", labelsize=7)
#        ax.set_ylabel("Weights", fontsize=8)
#        ax.set_xlabel("Bin Frequency", fontsize=8)
#    plt.tight_layout()
#
#    if plotSave == True:
#        plt.savefig(".//Plots//3_ROC_weigth_function.pdf", dpi=500, bbox_inches="tight")
    
    '''
    Plot 1: Different roc score under different lof nn parameters.
    '''
#    fig, axObj = plt.subplots(2, 3, figsize=(19, 8))
#    pts_index = [i for i in range(60, 200, 10)]
#    lw = 2
#    # Plot the best roc curve
#    axObj[0][0].plot(featureRocRec[2][featureRocBestInd][1],
#                     featureRocRec[2][featureRocBestInd][2],
#                     label='FeatureBased(auc={:.4f})'.format(featureRocBest),
#                     lw=lw, color="darkgreen")
#    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
#                         rocRec_ind[2][rocBestInd_ind][2],
#                         label= repName[ind] + '(auc={:.4f})'.format(rocBest_ind),
#                         lw=lw, color=colors[ind])
#    axObj[0][0].tick_params(axis="both", labelsize=10)
#    axObj[0][0].legend(fontsize=8)
#    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    
#    # Plot the remain roc curve
#    index_list = [0, 2, 4, 8, 13]
#    for ax_ind, (nn_pts, ax) in enumerate(zip(index_list, axObj.ravel()[1:])):
#        
#        pts = pts_index[nn_pts]
#        # Plot the feature-based roc curve
#        ax.plot(featureRocRec[2][nn_pts][1],
#                featureRocRec[2][nn_pts][2],
#                label='FeatureBased(nn=' + str(pts) + ', auc={:.3f})'.format(featureRocRec[1][nn_pts]),
#                lw=lw, color="darkgreen")
#        
#        # Plot the rep-based roc curve
#        for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
#            ax.plot(rocRec_ind[2][nn_pts][1],
#                    rocRec_ind[2][nn_pts][2],
#                    label= repName[ind] + '(auc={:.4f})'.format(rocRec_ind[1][nn_pts]),
#                    lw=lw, color=colors[ind])
#        ax.legend(fontsize=8)
#        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#        ax.tick_params(axis="y", labelsize=10)
#        ax.tick_params(axis="x", labelsize=10)
#    
#    for obj in axObj.ravel():
#        obj.grid(False)
#        obj.set_xlim(0, 1)
#        obj.set_ylim(0, )
#    plt.tight_layout()
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//3_ROC_roc_curve_different_rep.pdf", dpi=500, bbox_inches="tight")

    
    # Generating the LATEX format tables
    scoresCompare = pd.DataFrame(None)
    table_list = [60, 80, 100, 120, 140, 160, 180]
    scoresCompare["K"] = table_list
    scoresCompare["Features"] = [featureRocRec[1][ind] for ind, item in enumerate(featureRocRec[0]) if item in table_list]
    for ind, (name, score) in enumerate(zip(repName, rocRec)):
        scoresCompare[name] = [rocRec[ind][1][i] for i, item in enumerate(rocRec[ind][0]) if item in table_list]
    
    scoresCompare = scoresCompare[["K", "Features", "DAE-80", "SDAE-100-50", "GRU-40", "GRU-70",
                                   "Average-20-20", "Average-30-30", "Weighted-20-20",
                                   "Weighted-30-30"]]
    
    scores_str = ""
    with open(".//Models//" + '3_ROC_anomaly_detection_results.txt', 'w') as f:
        scores_vals = scoresCompare.values
        for i in range(scores_vals.shape[0]):
            for j in range(scores_vals.shape[1]):
                if j != (scores_vals.shape[1] - 1):
                    scores_str = scores_str + str(round(scores_vals[i, j], 3)) + " & "
                else:
                    scores_str = scores_str + str(round(scores_vals[i, j], 3)) + "\\\\"
            scores_str += "\n"
        f.write(scores_str)

    '''
    Plot 2: Weights changing, bins changing.
    '''
    plt.close("all")
    #####################################################
    fig, axObj = plt.subplots(2, 3, figsize=(19, 8))
    nn_nums = "in_3"
    for plot_ind, (ax, stride, windowSize) in enumerate(zip(axObj[0], [20, 30, 40], [20, 30, 40])):
        repName = [ind for ind, item in enumerate(fileNameAE) if ((str(windowSize) + "_" + str(stride)) in item) and (nn_nums in item)]
        print([item for ind, item in enumerate(fileNameAE) if ((str(windowSize) + "_" + str(stride)) in item) and (nn_nums in item)])
        print("\n")
        rep_tmp = [repOriginal[ind] for ind in repName][0]
        
        params = [[3.5, 10], [1.2, 10], [0.1, 10], [0.3, 10], [0.3, 25], [0.3, 50]]
        rep_test = [weighting_rep(rep_tmp, norm_factor=item[0], bin_name="bin_freq_" + str(item[1])) for item in params]
        rep_lof_results = [select_best_lof_value(rep_test[ind][1], y_true, [i for i in range(60, 200+20, 20)]) for ind in range(len(rep_test))]
        base_score = select_best_lof_value(rep_test[0][0], y_true, [i for i in range(60, 220, 20)])
        
        for ind, (res, param) in enumerate(zip(rep_lof_results, params)):
            ax.plot(res[2][0], res[2][1], lw=2,
                    color=colors[ind], marker=markers[ind],
                    label="alpha={}, bins={}".format(param[0], param[1]))
        ax.plot(base_score[2][0], base_score[2][1], lw=2,
                color="darkgreen", marker="x", linestyle="--",
                label="Simple average")
        ax.set_xlim(59, 201)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=8)
    print("Row 1 completed.")

    for plot_ind, (ax, stride) in enumerate(zip(axObj[1], [10, 20, 30])):
        repName = [ind for ind, item in enumerate(fileNameAE) if (("_40_" + str(stride)) in item) and (nn_nums in item)]
        print([item for ind, item in enumerate(fileNameAE) if (("_40_" + str(stride)) in item) and (nn_nums in item)])
        print("\n")
        rep_tmp = [repOriginal[ind] for ind in repName][0]
        
        params = [[3.5, 10], [1.2, 10], [0.1, 10], [0.3, 10], [0.3, 25], [0.3, 50]]
        rep_test = [weighting_rep(rep_tmp, norm_factor=item[0], bin_name="bin_freq_" + str(item[1])) for item in params]
        rep_lof_results = [select_best_lof_value(rep_test[ind][1], y_true, [i for i in range(60, 200+20, 20)]) for ind in range(len(rep_test))]
        base_score = select_best_lof_value(rep_test[0][0], y_true, [i for i in range(60, 220, 20)])
        
        for ind, (res, param) in enumerate(zip(rep_lof_results, params)):
            ax.plot(res[2][0], res[2][1], lw=2,
                    color=colors[ind], marker=markers[ind],
                    label="alpha={}, bins={}".format(param[0], param[1]))
        ax.plot(base_score[2][0], base_score[2][1], lw=2,
                color="darkgreen", marker="x", linestyle="--",
                label="Simple average")
        ax.set_xlim(59, 201)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    print("Row 2 completed.")
    
    if plotSave == True:
        plt.savefig(".//Plots//3_ROC_different_weights_bin_stride_windowSize_roc_"+ nn_nums + ".pdf", dpi=500, bbox_inches="tight")

    '''
    Plot 3: Increasing the neurons
    '''
#    plt.close("all")
#    #####################################################
#    fig, axObj = plt.subplots(2, 3, figsize=(19, 8))
#    
#    # Row 1
#    #--------------------------------------------------
#    # windowSize=(20, 30, 40), stride=(20, 30, 40)
#    for plot_ind, (ax, windowSize, stride) in enumerate(zip(axObj[0], [20, 30, 40], [20, 30, 40])):
#        repName_in = [ind for ind, item in enumerate(fileNameAE) if ((str(windowSize) + "_" + str(stride)) in item) and ("in" in item)]
#        print([item for ind, item in enumerate(fileNameAE) if ((str(windowSize) + "_" + str(stride)) in item) and ("in" in item)])
#
#        # in_0: 5    in_1: 15    in_2: 25    in_3:35    in_4:45    in_5:60
#        params = [0, 1, 2, 3, 4, 5]
#        rep_tmp_in = [repOriginal[item] for ind, item in enumerate(repName_in) if ind in params]
#        print([fileNameAE[item] for ind, item in enumerate(repName_in) if ind in params])
#        print("\n")
#        rep_test_in = [weighting_rep(item, norm_factor=0.1, bin_name="bin_freq_10") for item in rep_tmp_in]
#        rep_lof_results_in = [select_best_lof_value(rep_test_in[item][1], y_true, [i for i in range(60, 200+20, 20)]) for item in range(len(rep_test_in))]
#        for ind, (res, param) in enumerate(zip(rep_lof_results_in, [5, 15, 25, 35, 45, 60])):
#            ax.plot(res[2][0], res[2][1], lw=2,
#                    color=colors[ind], marker=markers[ind],
#                    label="Neurons={}".format(param))
#        ax.set_xlim(59, 201)
#        ax.tick_params(axis="both", labelsize=8)
#        ax.legend(fontsize=10)
#
#    # Row 2
#    #--------------------------------------------------
#    # windowSize=40, stride=(10, 20, 30)
#    for plot_ind, (ax, windowSize, stride) in enumerate(zip(axObj[1], [40, 40, 40], [10, 20, 30])):
#        repName_in = [ind for ind, item in enumerate(fileNameAE) if (str(windowSize) in item) and (str(stride) in item) and ("in" in item)]
#        print([item for ind, item in enumerate(fileNameAE) if ((str(windowSize) + "_" + str(stride)) in item) and ("in" in item)])
#
#        params = [0, 1, 2, 3, 4, 5]
#        rep_tmp_in = [repOriginal[item] for ind, item in enumerate(repName_in) if ind in params]
#        print([fileNameAE[item] for ind, item in enumerate(repName_in) if ind in params])
#        print("\n")
#        
#        rep_test_in = [weighting_rep(item, norm_factor=0.1, bin_name="bin_freq_10") for item in rep_tmp_in]
#        rep_lof_results_in = [select_best_lof_value(rep_test_in[item][1], y_true, [i for i in range(60, 200+20, 20)]) for item in range(len(rep_test_in))]
#        for ind, (res, param) in enumerate(zip(rep_lof_results_in, [5, 15, 25, 35, 45, 60])):
#            ax.plot(res[2][0], res[2][1], lw=2,
#                    color=colors[ind], marker=markers[ind],
#                    label="Neurons={}".format(param))
#        ax.set_xlim(59, 201)
#        ax.tick_params(axis="both", labelsize=8)
#        ax.legend(fontsize=10)
#
#    plt.tight_layout()
#    if plotSave == True:
#        plt.savefig(".//Plots//3_ROC_different_neurons_dropout_stride_windowSize_roc.pdf", dpi=500, bbox_inches="tight")

    '''
    Plot 5: ROC value boxenplot
    '''
#    plt.close("all")
#    fig, ax = plt.subplots(figsize=(12, 5))
#    roc_new = [item[1] for item in rocRec]
#    roc_new = roc_new + [featureRocRec[1]]
#    roc_new = pd.DataFrame(np.array(roc_new).T)
#    
#    ax = sns.boxenplot(data=roc_new, ax=ax)
#    ax.tick_params(axis="y", labelsize=8)
#    ax.tick_params(axis="x", labelsize=8)
#    ax.legend(fontsize=8)
#    
#    if plotSave == True:
#        plt.savefig(".//Plots//3_ROC_roc_boxplot.pdf", dpi=500, bbox_inches="tight")
    ###############################################
    ###############################################
    '''
    Plot 6: Anamoly detection results with the weights.
    '''
