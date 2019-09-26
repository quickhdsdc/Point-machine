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
#    ###############################################
#    ###############################################
#    '''
#    Step 1: Load all DAE representations, and perform the anamoly detection.
#    '''
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
#    load_all = True
#    if load_all:
#        featureRep = feature_kernel_pca(features=features.drop(["dateTime", "no"], axis=1), n_components=20)
#        rep, repOriginal, repRNN, repBase = [], [], [], []
#        for name in fileNameAE:
#            df = ls.load_data(PATH + name)
#            repOriginal.append(df.copy())
#            rep.append(weighting_rep(df, norm_factor=0.05, bin_name="bin_freq_10"))
#            
#        for name in fileNameRNN:
#            repRNN.append(ls.load_data(PATH + name))
#        for name in fileNameBase:
#            repBase.append(ls.load_data(PATH + name))
#            repBase[-1] = repBase[-1][[name for name in repBase[-1].columns if "rep" in name]].values
    ###############################################
    ###############################################
    # Normal samples index
    normal_index = groundTruth[groundTruth["label"] != -1]["ind"].values.astype("int")
    normal_index_label = groundTruth[groundTruth["label"] != -1]["label"].values.astype("int")
    
    # Anamoly detection: select the best score
    # Step ==> 10, 20, 30, neurons(20), windowSize(40)
    #----------------------------------------------------
    repList = [rep[19][0],  rep[32][0], repRNN[3], repRNN[0],
               repBase[0], repBase[1], rep[19][1],  rep[32][1]]
    
    # Scaling the features
    featureTmp = features.copy()
    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
    X_sc = StandardScaler()
    for name in featureTmp.columns:
        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))    

    ###############################################
    ###############################################
#    # Start the clustering experiments
#    v_score, ami = [], []
#    cluster_seq = [i for i in range(2, 25+3, 3)]
#    random_state = 9102
#    
#    for c_nums in cluster_seq:
#        print("\n---------------c_nums={}-------------------".format(c_nums))
#        scores_v_score = np.zeros((3, len(repList) + 1))
#        scores_ami = np.zeros((3, len(repList) + 1))
#        
#        # KMeans clustering score
#        #----------------------------------------------------
#        clf_kmeans = KMeans(n_clusters=c_nums, random_state=random_state)
#        
#        X_sc = StandardScaler()
#        for ind, ts_rep in enumerate(repList):
#            print("-- KMeans now is {}.".format(ind))
#            tmp = X_sc.fit_transform(ts_rep)
#            clf_kmeans.fit(tmp)
#            scores_v_score[0, ind] = v_measure_score(normal_index_label, clf_kmeans.labels_[normal_index])
#            scores_ami[0, ind] = adjusted_mutual_info_score(normal_index_label, clf_kmeans.labels_[normal_index])
#            
#        clf_kmeans.fit(featureTmp.values)
#        scores_v_score[0, -1] = v_measure_score(normal_index_label, clf_kmeans.labels_[normal_index])
#        scores_ami[0, -1] = adjusted_mutual_info_score(normal_index_label, clf_kmeans.labels_[normal_index])
#        
#        # Spectral clustering score
#        #----------------------------------------------------
#        for ind, ts_rep in enumerate(repList):
#            print("-- SC now is {}.".format(ind))
#            dist_mat = pairwise_distances(ts_rep, metric="euclidean")
#            dist_mat = np.exp( np.divide(-np.square(dist_mat), 2 * ts_rep.shape[1] ** 2) )
#            
#            clf_sc = SpectralClustering(n_clusters=c_nums,
#                                        random_state=random_state,
#                                        affinity="precomputed")
#            clf_sc.fit(dist_mat)
#            scores_v_score[1, ind] = v_measure_score(normal_index_label, clf_sc.labels_[normal_index])
#            scores_ami[1, ind] = adjusted_mutual_info_score(normal_index_label, clf_sc.labels_[normal_index])
#            gc.collect()
#            
#        dist_mat = pairwise_distances(featureTmp.values, metric="euclidean")
#        dist_mat = np.exp( np.divide(-np.square(dist_mat), 2 * featureTmp.shape[1] ** 2) )
#        clf_sc.fit(dist_mat)
#        scores_v_score[0, -1] = v_measure_score(normal_index_label, clf_sc.labels_[normal_index])
#        scores_ami[0, -1] = adjusted_mutual_info_score(normal_index_label, clf_sc.labels_[normal_index])
#        
#        # Agglomerative clustering score
#        #----------------------------------------------------
#        X_sc = StandardScaler()
#        for ind, ts_rep in enumerate(repList):
#            print("-- Agglomerative Clustering now is {}.".format(ind))
#            tmp = X_sc.fit_transform(ts_rep)
#            clf_ag = AgglomerativeClustering(n_clusters=c_nums)
#            clf_ag.fit(tmp)
#            scores_v_score[2, ind] = v_measure_score(normal_index_label, clf_ag.labels_[normal_index])
#            scores_ami[2, ind] = adjusted_mutual_info_score(normal_index_label, clf_ag.labels_[normal_index])
#            
#        clf_ag.fit(featureTmp.values)
#        scores_v_score[2, -1] = v_measure_score(normal_index_label, clf_ag.labels_[normal_index])
#        scores_ami[2, -1] = adjusted_mutual_info_score(normal_index_label, clf_ag.labels_[normal_index])
#        
#        v_score.append(pd.DataFrame(scores_v_score, columns=["averge_0", "average_1",
#                                                             "gru_0", "gru_1",
#                                                             "DAE", "SAE",
#                                                             "weighted_0", "weighted_1",
#                                                             "feature_based"]))
#        ami.append(pd.DataFrame(scores_ami, columns=["averge_0", "average_1",
#                                                     "gru_0", "gru_1",
#                                                     "DAE", "SAE",
#                                                     "weighted_0", "weighted_1",
#                                                     "feature_based"]))
    ###############################################
    ###############################################
    repName = ["averge_0", "average_1", "gru_0", "gru_1",
               "DAE", "SAE", "weighted_0", "weighted_1"]
    lw = 2
    
    plt.close("all")
    fig, axObj = plt.subplots(1, 2, figsize=(14, 5))
    for ind, (score, col_name) in enumerate(zip(v_score, repName)):
        all_scores_v = [df[col_name].loc[0] for df in v_score]
        all_scores_a = [df[col_name].loc[0] for df in ami]
        
        axObj[0].plot(cluster_seq, all_scores_v, color=colors[ind],
             marker=markers[ind], lw=lw, label=col_name)
        axObj[1].plot(cluster_seq, all_scores_a, color=colors[ind],
             marker=markers[ind], lw=lw, label=col_name)
        
        axObj[0].set_xlim(1, max(cluster_seq)+1)
        axObj[0].tick_params(axis="both", labelsize=8)
        axObj[0].legend(fontsize=8)

        axObj[1].set_xlim(1, max(cluster_seq)+1)
        axObj[1].tick_params(axis="both", labelsize=8)
        axObj[1].legend(fontsize=8)
        
    plt.tight_layout()
    if plotSave == True:
        plt.savefig(".//Plots//4_CLUSTERING_clustering_score_kmeans.png", dpi=500, bbox_inches="tight")    