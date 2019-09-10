#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:57:03 2019

@author: yinzhuo
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from functools import wraps
import pandas as pd
import gc
from timeit import timeit
from tsfresh.feature_extraction import feature_calculators as fc
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn import manifold
import warnings
warnings.filterwarnings('ignore')

from UTILS import ReduceMemoryUsage, LoadSave, basic_feature_report, plot_single_record
np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
mpl.rcParams["font.family"] = "Times New Roman"
###############################################################################
###############################################################################
def plot_random_n_current(n=100, seq=None):
    numData = len(seq)
    randIndex = np.random.randint(0, numData, n)
    
    plt.figure()
    for ind, item in enumerate(randIndex):
        plt.plot(seq[item], color="blue", linewidth=0.5, linestyle="-")
        
        
if __name__ == "__main__":
    plotShow, plotSave = False, True
    colors = ["C" + str(i) for i in range(0, 9+1)]
    
    ls = LoadSave()
    data = ls.load_data(".//Data//current_ts_merged_3300.pkl")
    groutdTruth, ts, features = data[0].drop(["centerInd"], axis=1), data[1], data[2]
    ###############################################################################
    ###############################################################################
    plt.close("all")
    '''
    Step 1: Plot random n signals, seems there are only 1 class in this set
    '''
#    f, ax = plt.subplots(figsize=(8, 6))
#    index = np.random.randint(0, len(ts), 10)
#    for ind, item in enumerate(index):
#        ax.plot(ts[item], color="b", linewidth=1.3, linestyle="-")
#    ax.tick_params(axis="both", labelsize=8)
#    ax.set_xlim(0, max([len(ts[i]) for i in index]))
#    if plotSave:
#        plt.savefig(".//Plots//random_n_signal.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
#    # Some basic preprocessing
#    report = basic_feature_report(features)
#    dropList = list(report.query("nuniqueValues <= 1")["featureName"].values)
#    features.drop(dropList, inplace=True, axis=1)
#    report = basic_feature_report(features)
#    print("Number of dropped features:{}".format(len(dropList)))
#    features.fillna(0, axis=1, inplace=True)
    
    '''
    Step 2: Heatmap of all features.
    '''    
    # Features heatmap
    featureTmp = features.copy()
    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    featureCorr = featureTmp.corr()
#    f, ax = plt.subplots(figsize=(16, 12))
#    ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
#                     cmap="Blues", fmt='.2f', annot=True,
#                     annot_kws={'size':4.5,'weight':'bold'}, ax=ax)
##    ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1, cmap="Blues", ax=ax)
#    plt.xticks(rotation=90)
#    plt.yticks(rotation=0)
#    
#    # Adjasting the ticks fontsize
#    ax.tick_params(axis="y", labelsize=8)
#    ax.tick_params(axis="x", labelsize=8)
#    
#    # Adjasting the colorbar size
#    cbar = ax.collections[0].colorbar
#    cbar.ax.tick_params(labelsize=10)
#    
#    if plotSave:
#        plt.savefig(".//Plots//heatmap_features.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")

    # Boxplot for the features: max, min, median, mean, absEnergy
#    f, ax = plt.subplots(figsize=(10, 6))
#    ax.boxplot(features[["mean", "median", "max", "var", "meanAbsChange", "segVar_1"]].values)
#    ax.set_xticklabels(["mean", "median", "max", "var", "meanAbsChange", "segVar_1"])
#    ax.tick_params(axis="x", labelsize=8)
##    sns.boxplot(data=features[["mean", "median", "max", "var", "meanAbsChange", "segVar_1"]], color='.25', palette='deep', linewidth=2.5)
#    if plotSave:
#        plt.savefig(".//Plots//boxplot_features.png", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 3: PCA plot of features.
    '''
    X_sc = StandardScaler()
    for name in featureTmp.columns:
        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
    
    components = 17
    pca = PCA(n_components=components)
    featuresPCA = pca.fit_transform(featureTmp.values)
    varRemain = pca.explained_variance_ratio_
    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
    print("@Information reamins : {}".format(sum(varRemain)))
    
    clf, clfNoiseLevel, clfNoiseLabel = [], [], []
    clf_0 = LocalOutlierFactor(n_neighbors=8, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_0 = clf_0.fit_predict(featuresPCA)
    noiseLevel_0 = clf_0.negative_outlier_factor_

    clf_1 = LocalOutlierFactor(n_neighbors=10, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_1 = clf_1.fit_predict(featuresPCA)
    noiseLevel_1 = clf_1.negative_outlier_factor_

    clf_2 = LocalOutlierFactor(n_neighbors=100, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_2 = clf_2.fit_predict(featuresPCA)
    noiseLevel_2 = clf_2.negative_outlier_factor_
    
    clf_3 = LocalOutlierFactor(n_neighbors=300, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_3 = clf_3.fit_predict(featuresPCA)
    noiseLevel_3 = clf_3.negative_outlier_factor_
    
    clf_4 = LocalOutlierFactor(n_neighbors=500, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_4 = clf_4.fit_predict(featuresPCA)
    noiseLevel_4 = clf_4.negative_outlier_factor_

    clf_5 = LocalOutlierFactor(n_neighbors=1000, metric="euclidean",
                               contamination=0.08, n_jobs=1)
    noiseLabels_5 = clf_5.fit_predict(featuresPCA)
    noiseLevel_5 = clf_5.negative_outlier_factor_
    
    clf = [clf_0, clf_1, clf_2, clf_3, clf_4, clf_5]
    clfNoiseLevel = [noiseLevel_0, noiseLevel_1, noiseLevel_2,
                     noiseLevel_3, noiseLevel_4, noiseLevel_5]
    clfNoiseLabel = [noiseLabels_0, noiseLabels_1, noiseLabels_2,
                     noiseLabels_3, noiseLabels_4, noiseLabels_5]
    
    components = 2
    pca = PCA(n_components=components)
    featuresPCA = pca.fit_transform(featureTmp.values)
    varRemain = pca.explained_variance_ratio_
    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
    print("@Information reamins : {}".format(sum(varRemain)))
    
    # Factor color plot
    f, axObj = plt.subplots(2, 3, figsize=(16, 9))
    for ax, noiseLevel, noiseLabels in zip(axObj.ravel(), clfNoiseLevel, clfNoiseLabel):
        ax.scatter(featuresPCA[:, 0], featuresPCA[:, 1], c="b",
                   marker=".", label="Normal")
        area = (- noiseLevel[noiseLabels == -1]) * 20
        ax.scatter(featuresPCA[:, 0][noiseLabels == -1], featuresPCA[:, 1][noiseLabels == -1], s=area, c="r", alpha=0.5)
#        ax.grid(True)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()

    if plotSave:
        plt.savefig(".//Plots//pca_features_area.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
        
    # Non-color plot
    f, axObj = plt.subplots(2, 3, figsize=(16, 9))
    for ax, noiseLevel, noiseLabels in zip(axObj.ravel(), clfNoiseLevel, clfNoiseLabel):
        ax.scatter(featuresPCA[:, 0][noiseLabels == 1],
                   featuresPCA[:, 1][noiseLabels == 1], color="b", marker=".", label="Normal")
#        ax.grid(True)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//pca_features.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
    
    # Colored by the ground truth
    f, ax = plt.subplots(figsize=(8, 6))
    uniqueLabels = -np.sort(-groutdTruth["label"].unique())
    for ind, label in enumerate(uniqueLabels):
        sampleIndex = np.arange(0, len(groutdTruth))
        labeledSampleIndex = sampleIndex[groutdTruth["label"] == label]
        
        coords = featuresPCA[labeledSampleIndex, :]
        if label != -1:
            ax.scatter(coords[:, 0], coords[:, 1], s=10, 
                       color='b', marker=".")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=10, 
                       color='r', marker="x", label="Class Abnormal")    
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
    plt.legend(fontsize=10)

    if plotSave:
        plt.savefig(".//Plots//pca_2d_colored_by_truth.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
    
    '''
    Step 4: Date time based features visualizing.
    '''
    featuresTmp = features[["length", "mean", "segMean_2", "dateTime"]].copy()
    featuresTmp["day"] = featuresTmp["dateTime"].dt.day
    featuresTmp["hour"] = featuresTmp["dateTime"].dt.hour
    featuresTmp["month"] = featuresTmp["dateTime"].dt.month
    
    # Hour length plot
    f, axObj = plt.subplots(2, 1, figsize=(10, 8))
    ax = sns.axObj[0](x="hour", y="length", data=featuresTmp, color="b", ax=axObj[0])
    axObj[0].tick_params(axis="y", labelsize=10)
    axObj[0].tick_params(axis="x", labelsize=10)
    axObj[0].set_xlabel("hour", fontsize=10)
    axObj[0].set_ylabel("length", fontsize=10)
    
    # Hour length plot
    lengthHourTmp = featuresTmp.groupby(["hour"])["length"].agg(["mean", "std"]).reset_index()
    ax = sns.catplot(x="hour", y="length", capsize=0.2, palette="YlGnBu_d",
                     height=5, kind="point", data=featuresTmp, ax=axObj[1])
    axObj[1].tick_params(axis="y", labelsize=10)
    axObj[1].tick_params(axis="x", labelsize=10)
    axObj[1].set_xlabel("hour", fontsize=10)
    axObj[1].set_ylabel("length", fontsize=10)
    plt.tight_layout()
    
    if plotSave:
        f.savefig(".//Plots//catplot.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
    
    '''
    Step 5: Find the abnormal seq in segment_1.
    '''
    df = featureTmp[["segVar_1", "segMean_1", "segSlope_1"]].values
    X_sc = StandardScaler()
    df = X_sc.fit_transform(df)
    
    clf = DBSCAN(eps=0.3, min_samples=30)
    labels = clf.fit_predict(df)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    ax.scatter(df[:, 0][labels != -1], df[:, 1][labels != -1],
               df[:, 2][labels != -1], marker=".", color="k", s=30)
    ax.scatter(df[:, 0][labels == -1], df[:, 1][labels == -1],
               df[:, 2][labels == -1], marker="x", color="r", s=30)
    
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="z", labelsize=10)

    ax.set_xlabel('segVar_1', fontdict={'size': 13}, color="g")
    ax.set_ylabel('segMean_1', fontdict={'size': 13}, color="g")
    ax.set_zlabel('segSlope_1', fontdict={'size': 13}, color="g")
    plt.tight_layout()

    if plotSave:
        plt.savefig(".//Plots//seg_features_3d.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
    
    '''
    
    Step 6: T-SNE visvalizing.
    '''
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#        
#    # TSNE visualizing
#    from matplotlib import offsetbox
#    def plot_embedding(X, y, title=None):
#        x_min, x_max = np.min(X, 0), np.max(X, 0)
#        X = (X - x_min) / (x_max - x_min)
#    
#        plt.figure()
#        ax = plt.subplot(111)
#        for i in range(X.shape[0]):
#            plt.scatter(X[i, 0], X[i, 1], c=plt.cm.Set1(y[i] / 10))
#    
#        if title is not None:
#            plt.title(title)
#    tsne = manifold.TSNE(n_components=2, init='pca', random_state=2019)
#    X_tsne = tsne.fit_transform(featureTmp.values)
#    plot_embedding(X_tsne, groutdTruth["label"].values)
    