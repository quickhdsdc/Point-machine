#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:53:29 2019

@author: michaelyin1994
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
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score, pairwise_distances
from tqdm import tqdm

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
def clustering_raw_features(features=[], n_clusters=10, method="spectral"):
    if (len(features) == 0) or (n_clusters < 2):
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))
        
    # Step 3: Start clustering using the compressed data
    if method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, n_init=20, gamma=0.5,
                                 affinity="rbf", n_jobs=-1)
    else:    
        clf = KMeans(n_clusters=n_clusters, n_jobs=-1)
    labels = clf.fit_predict(features.values)
    return labels


def clustering_pca_features(features=[], n_clusters=10, n_components=10, method="spectral"):
    if (len(features) == 0) or (n_clusters < 2) or (n_components < 2):
        return None
    
    # Step 1: Standardizing the input data
    X_sc = StandardScaler()
    for name in features.columns:
        features[name].fillna(features[name].mean(), inplace=True)
        features[name] = X_sc.fit_transform(features[name].values.reshape(len(features), 1))
        
    # Step 2: Get the features after the PCA processing
    pca = PCA(n_components=n_components)
    featuresPCA = pca.fit_transform(features.values)
    varRemain = pca.explained_variance_ratio_
    print("Information reamins : {}".format(sum(varRemain)))
    
    # Step 3: Start clustering using the compressed data
    if method == "spectral":
        clf = SpectralClustering(n_clusters=n_clusters, n_init=20, gamma=0.5,
                                 affinity="rbf", n_jobs=-1)
    else:    
        clf = KMeans(n_clusters=n_clusters, n_jobs=-1)
    labels = clf.fit_predict(features.values)
    labels = clf.fit_predict(featuresPCA)
    return labels


def clustering_dtw_spectral(data=[], n_clusters=10):
    
    pass


def clustering_fastdtw_spectral(data=[], n_clusters=10):
    
    pass


def clustering_dae():
    pass


def clustering_ae():
    pass


def plot_clustering_results(data=[], c_label=[], t_label=[], methodStr="..//Plots//FeaturePcaBased//FeaturePcaBased_", plotNums=100):
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
        for i in range(plotNums):
            plotList.append(random.choice(indexList))
        
        # Start plot
        f, ax = plt.subplots()
        for i in plotList:
            signal = data.iloc[i]["current"]
            plt.plot(signal, color="blue", linewidth=1.2, linestyle="-")
        plt.xlabel("time(s)")
        plt.ylabel("current(A)")
        plt.title("Cluster " + str(label))
        plt.savefig(methodStr + str(label) + ".png", dpi=500, bbox_inches="tight")
        plt.close("all")


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
#    PATH = ".//Data//rep_data//"
#    ls = LoadSave(PATH)
#    fileName = os.listdir(PATH)
#    
#    # Get the representations
#    fileNameAE = sorted([name for name in fileName if ("rnn" not in name) and ("base" not in name)])
#    fileNameRNN = sorted([name for name in fileName if "rnn" in name])
#    fileNameBase = sorted([name for name in fileName if "base" in name])
#    
#    # Get the representations
#    load_all = False
#    if load_all:
#        featureRep = feature_kernel_pca(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
#        rep, repOriginal, repRNN, repBase = [], [], [], []
#        for name in fileNameAE:
#            df = ls.load_data(PATH + name)
#            repOriginal.append(df.copy())
#            rep.append(weighting_rep(df, norm_factor=0.1, bin_name="bin_freq_10"))
#            
#        for name in fileNameRNN:
#            repRNN.append(ls.load_data(PATH + name))
#        for name in fileNameBase:
#            repBase.append(ls.load_data(PATH + name))
#            repBase[-1] = repBase[-1][[name for name in repBase[-1].columns if "rep" in name]].values
    ###############################################
    ###############################################
#    # Normal samples index
#    normal_index = groundTruth[groundTruth["label"] != -1]["ind"].values.astype("int")
#    normal_index_label = groundTruth[groundTruth["label"] != -1]["label"].values.astype("int")
#    
#    # Scaling the features
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))    
#
#    # Anamoly detection: select the best score
#    # Step ==> 10, 20, 30, neurons(20), windowSize(40)
#    #----------------------------------------------------
#    repList = [featureTmp, rep[19][0],  rep[32][0], repRNN[0], repRNN[1],
#               repBase[0], repBase[1], rep[19][1],  rep[32][1]]
#    repName = ["FeatureBased", "Average-20-20", "Average-30-30", "GRU-40", "GRU-70",
#               "DAE-80", "SDAE-100-50", "Weighted-20-20", "Weighted-30-30"]
    
#    repList = [featureTmp, rep[20][0],  rep[21][0], repRNN[0], repRNN[1],
#               repBase[0], repBase[1], rep[20][1],  rep[21][1]]
#    repName = ["FeatureBased", "Average-40-10", "Average-40-20", "GRU-40", "GRU-70",
#               "DAE-80", "SDAE-100-50", "Weighted-40-10", "Weighted-40-20"]    

    ###############################################
    ###############################################
    # Start the clustering experiments
#    v_score = {}
#    cluster_seq = [i for i in range(7, 30+3, 3)]
#    random_state_seq = [np.random.randint(1000) + i for i in range(len(cluster_seq))]
#    
#    # Start the clustering experiments    
#    for c_nums in cluster_seq:
#        print("\n---------------c_nums={}-------------------".format(c_nums))
#        v_score_kmeans, v_score_agg = {}, {}
#        
#        # KMeans clustering score
#        #----------------------------------------------------
#        X_sc = StandardScaler()
#        for ind_rep, (ts_rep, ts_name) in enumerate(zip(repList, repName)):
#            print("-- KMeans now is {}, total is {}.".format(ind_rep+1, len(repList)))
#            ts_rep_scaled = X_sc.fit_transform(ts_rep)
#            score_tmp = []
#            
#            for ind_seed, seed in enumerate(random_state_seq):
#                clf_kmeans = KMeans(n_clusters=c_nums, random_state=seed, n_jobs=1)
#                clf_kmeans.fit(ts_rep_scaled)
#                
#                score_tmp.append(v_measure_score(normal_index_label, clf_kmeans.labels_[normal_index]))
#            v_score_kmeans[ts_name] = np.mean(score_tmp)
#
#        # Agglomerative clustering score
#        #----------------------------------------------------
#        X_sc = StandardScaler()
#        for ind_rep, (ts_rep, ts_name) in enumerate(zip(repList, repName)):
#            print("-- Agglomerative Clustering now is {}, total is {}.".format(ind_rep+1, len(repList)))
#            ts_rep_scaled = X_sc.fit_transform(ts_rep)
#            score_tmp = None
#            
#            clf_ag = AgglomerativeClustering(n_clusters=c_nums)
#            clf_ag.fit(ts_rep_scaled)
#            score_tmp = v_measure_score(normal_index_label, clf_ag.labels_[normal_index])
#            v_score_agg[ts_name] = score_tmp
#            
#        v_score[c_nums] = [v_score_kmeans, v_score_agg]
#    
#    # Rebuild to the data frame
#    repName, cluster_seq = repName, cluster_seq
#    scores_kmeans, scores_agg = [v_score[item][0] for ind, item in enumerate(v_score.keys())], [v_score[item][1] for ind, item in enumerate(v_score.keys())]
#    heat_kmeans, heat_agg = [list(scores_kmeans[ind].values()) for ind, item in enumerate(scores_kmeans)], [list(scores_agg[ind].values()) for ind, item in enumerate(scores_agg)]
#    heat_kmeans, heat_agg = np.array(heat_kmeans).T, np.array(heat_agg).T
    
    ###############################################
    ###############################################
    '''
    Plot 1: clustering scores heatmap
    '''
#    heat_kmeans_with_mean = np.hstack([heat_kmeans, np.mean(heat_kmeans, axis=1).reshape(-1, 1)])
#    heat_agg_with_mean = np.hstack([heat_agg, np.mean(heat_agg, axis=1).reshape(-1, 1)])
#    cluster_seq_with_mean = cluster_seq + ["mean"]
#    
#    fig, axObj = plt.subplots(1, 2, figsize=(12, 4))
#    for ind, (ax, featureCorr) in enumerate(zip(axObj, [heat_kmeans_with_mean, heat_agg_with_mean])):
#        ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
#                         cmap="Blues", fmt='.3f', annot=True,
#                         annot_kws={'size':5.5,'weight':'bold'}, ax=ax)
#
#        ax.set_xticklabels(cluster_seq_with_mean, rotation=0)
#        ax.set_yticklabels(repName, rotation=0)
#        ax.tick_params(axis="y", labelsize=7, rotation=0)
#        ax.tick_params(axis="x", labelsize=7)
#        
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=7)
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//4_CLUSTERING_kmeans_agg_v_measure_heatmap.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
        
    ###############################################
    ###############################################
    v_score = {}
    cluster_seq = [i for i in range(7, 30+3, 3)]
    random_state_seq = [np.random.randint(1000) + i for i in range(len(cluster_seq))]

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
        rep_list = [weighting_rep(rep_tmp, norm_factor=item[0], bin_name="bin_freq_" + str(item[1])) for item in params]
        
        
        
        
        
#        for ind, (res, param) in enumerate(zip(rep_lof_results, params)):
#            ax.plot(res[2][0], res[2][1], lw=2,
#                    color=colors[ind], marker=markers[ind],
#                    label="alpha={}, bins={}".format(param[0], param[1]))
#        ax.plot(base_score[2][0], base_score[2][1], lw=2,
#                color="darkgreen", marker="x", linestyle="--",
#                label="Simple average")
#        ax.set_xlim(59, 201)
#        ax.tick_params(axis="both", labelsize=10)
#        ax.legend(fontsize=8)
#    print("Row 1 completed.")
#
#    for plot_ind, (ax, stride) in enumerate(zip(axObj[1], [10, 20, 30])):
#        repName = [ind for ind, item in enumerate(fileNameAE) if (("_40_" + str(stride)) in item) and (nn_nums in item)]
#        print([item for ind, item in enumerate(fileNameAE) if (("_40_" + str(stride)) in item) and (nn_nums in item)])
#        print("\n")
#        rep_tmp = [repOriginal[ind] for ind in repName][0]
#        
#        params = [[3.5, 10], [1.2, 10], [0.1, 10], [0.3, 10], [0.3, 25], [0.3, 50]]
#        rep_test = [weighting_rep(rep_tmp, norm_factor=item[0], bin_name="bin_freq_" + str(item[1])) for item in params]
#        rep_lof_results = [select_best_lof_value(rep_test[ind][1], y_true, [i for i in range(60, 200+20, 20)]) for ind in range(len(rep_test))]
#        base_score = select_best_lof_value(rep_test[0][0], y_true, [i for i in range(60, 220, 20)])
#        
#        for ind, (res, param) in enumerate(zip(rep_lof_results, params)):
#            ax.plot(res[2][0], res[2][1], lw=2,
#                    color=colors[ind], marker=markers[ind],
#                    label="alpha={}, bins={}".format(param[0], param[1]))
#        ax.plot(base_score[2][0], base_score[2][1], lw=2,
#                color="darkgreen", marker="x", linestyle="--",
#                label="Simple average")
#        ax.set_xlim(59, 201)
#        ax.tick_params(axis="both", labelsize=10)
#        ax.legend(fontsize=8)
#    plt.tight_layout()
#    print("Row 2 completed.")
    
    if plotSave == True:
        plt.savefig(".//Plots//4_CLUSTERING_bin_stride_changing_"+ nn_nums + ".pdf", dpi=500, bbox_inches="tight")    
    
    

    