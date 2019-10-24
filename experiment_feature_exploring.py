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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import roc_curve, auc, v_measure_score, pairwise_distances
import warnings
warnings.filterwarnings('ignore')

from UTILS import ReduceMemoryUsage, LoadSave, basic_feature_report, plot_single_record
np.random.seed(2019)
markers = ["s", "^", "o", "d", "*", "<", ">", "h", "p"]
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
    groundTruth, ts, features = data[0].drop(["centerInd"], axis=1), data[1], data[2]
    ###############################################################################
    ###############################################################################
    plt.close("all")
    
    '''
    Step -1: Try some new plots.
    '''
#    components = 2
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#    pca = KernelPCA(n_components=components, kernel='rbf')
#    featuresPCA = pca.fit_transform(featureTmp.values)
#    varRemain = pca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))    
#    
#    g = sns.jointplot(x=featuresPCA[:, 0], y=featuresPCA[:, 1], kind="kde", color="b")
#    g.plot_joint(plt.scatter, c="r", s=30, linewidth=1, marker="+")
#    g.ax_joint.collections[0].set_alpha(0)
#    g.set_axis_labels("$X$", "$Y$")
    
    '''
    Step 0: Plot 4 typical noisy signal.
    '''
#    fig, axObj = plt.subplots(1, 4, figsize=(16, 3))
#    noiseInd = [i for i in list(features.index) if groundTruth["label"].loc[i] == -1]
#    noiseTs = [ts[i] for i in noiseInd]
#    noiseFeatures = features[groundTruth["label"] == -1]
#    # Noise 0: Almost zero mean.
#    # Noise 1: 
#    
#    noiseInd_0, noiseInd_1, noiseInd_2, noiseInd_3 = 2421, 1648, 321, 260
#    allNoise = [noiseInd_0, noiseInd_1, noiseInd_2, noiseInd_3]
#    
#    for ax, ind in zip(axObj.ravel(), allNoise):
#        ax.plot(ts[ind], linewidth=2, color="k")
#        ax.tick_params(axis="both", labelsize=8)
#        ax.set_ylim(0, 7)
#        ax.set_xlim(0, len(ts[ind]))
#    plt.tight_layout()
#
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_plot_4_fault.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 1: Plot random n signals, seems there are only 3 class in this set
    '''
#    fig, axObj = plt.subplots(1, 4, figsize=(16, 3))
#    labelToPlot, signalsEachPlot = [1, 2, 3, -1], 15
#    selectedSamples = []
#    for label in labelToPlot:
#        sampleInd = groundTruth[groundTruth["label"] == label]["ind"].values
#        selectedSamples.append(np.random.choice(sampleInd, size=signalsEachPlot, replace=False))
#    
#    # Plot the original signals
#    for ind, item in enumerate(axObj):
#        sampleInd = selectedSamples[ind]
#        for i in sampleInd:
#            item.plot(ts[int(i)], color="k", linewidth=1.5)
#            item.tick_params(axis="y", labelsize=8)
#            item.tick_params(axis="x", labelsize=8)
#            item.set_xlim(0, max([len(ts[int(i)]) for i in sampleInd]))
#            item.set_ylim(0, 7)
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_plot_random_n_signal.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
            
    '''
    Step 2: Heatmap of all features.
    '''    
#    # Features heatmap
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
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
#        plt.savefig(".//Plots//0_EDA_heatmap_original_features.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")

    '''
    Step 3: Boxplot for the features: max, min, median, mean, absEnergy.
    '''
#    f, ax = plt.subplots(figsize=(10, 6))
#    flierprops = dict(marker='o', markersize=3)
#    sns.boxplot(data=features[["mean", "median", "max", "var", "meanAbsChange", "segVar_1"]],
#                color='b', linewidth=1.5, fliersize=3, flierprops=flierprops)
#    ax.tick_params(axis="both", labelsize=8)
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_boxplot_features.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 4: Kernel PCA plot of features, and using the LOF to detect the outliers.
    '''
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#    
#    # kernel pca
#    components = 17
#    pca = KernelPCA(n_components=components)
#    featuresPCA = pca.fit_transform(featureTmp.values)
#    varRemain = pca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    clf, clfNoiseLevel, clfNoiseLabel = [], [], []
#    clf_0 = LocalOutlierFactor(n_neighbors=8, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_0 = clf_0.fit_predict(featuresPCA)
#    noiseLevel_0 = clf_0.negative_outlier_factor_
#
#    clf_1 = LocalOutlierFactor(n_neighbors=40, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_1 = clf_1.fit_predict(featuresPCA)
#    noiseLevel_1 = clf_1.negative_outlier_factor_
#
#    clf_2 = LocalOutlierFactor(n_neighbors=80, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_2 = clf_2.fit_predict(featuresPCA)
#    noiseLevel_2 = clf_2.negative_outlier_factor_
#    
#    clf_3 = LocalOutlierFactor(n_neighbors=150, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_3 = clf_3.fit_predict(featuresPCA)
#    noiseLevel_3 = clf_3.negative_outlier_factor_
#    
#    clf_4 = LocalOutlierFactor(n_neighbors=300, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_4 = clf_4.fit_predict(featuresPCA)
#    noiseLevel_4 = clf_4.negative_outlier_factor_
#
#    clf_5 = LocalOutlierFactor(n_neighbors=500, metric="euclidean",
#                               contamination=0.08, n_jobs=1)
#    noiseLabels_5 = clf_5.fit_predict(featuresPCA)
#    noiseLevel_5 = clf_5.negative_outlier_factor_
#    
#    clf = [clf_0, clf_1, clf_2, clf_3, clf_4, clf_5]
#    clfNoiseLevel = [noiseLevel_0, noiseLevel_1, noiseLevel_2,
#                     noiseLevel_3, noiseLevel_4, noiseLevel_5]
#    clfNoiseLabel = [noiseLabels_0, noiseLabels_1, noiseLabels_2,
#                     noiseLabels_3, noiseLabels_4, noiseLabels_5]
#
#    # kpca ===>> 2 components
#    components = 2
#    pca = KernelPCA(n_components=components, kernel='rbf')
#    featuresPCA = pca.fit_transform(featureTmp.values)
#    varRemain = pca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # Factor color plot
#    f, axObj = plt.subplots(2, 3, figsize=(16, 9))
#    for ax, noiseLevel, noiseLabels in zip(axObj.ravel(), clfNoiseLevel, clfNoiseLabel):
#        ax.scatter(featuresPCA[:, 0], featuresPCA[:, 1], c="b",
#                   marker=".", label="Normal")
#        area = (- noiseLevel[noiseLabels == -1]) * 20
#        ax.scatter(featuresPCA[:, 0][noiseLabels == -1], featuresPCA[:, 1][noiseLabels == -1], s=area, c="r", alpha=0.5)
##        ax.grid(True)
#        ax.tick_params(axis="y", labelsize=10)
#        ax.tick_params(axis="x", labelsize=10)
#    plt.tight_layout()
#
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_pca_features_area.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#
#    # xxxx plot
#    f, axObj = plt.subplots(2, 3, figsize=(16, 9))
#    for ax, noiseLevel, noiseLabels in zip(axObj.ravel(), clfNoiseLevel, clfNoiseLabel):
#        ax.scatter(featuresPCA[:, 0][noiseLabels != -1], featuresPCA[:, 1][noiseLabels != -1], c="b",
#                   marker=".", s=15, label="Normal")
#        area = (- noiseLevel[noiseLabels == -1]) * 20
#        ax.scatter(featuresPCA[:, 0][noiseLabels == -1], featuresPCA[:, 1][noiseLabels == -1], s=15, c="r", marker="x")
#        ax.tick_params(axis="y", labelsize=10)
#        ax.tick_params(axis="x", labelsize=10)
#    plt.tight_layout()
#
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_pca_features_xxx.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#        
#    # Non-color plot
#    f, axObj = plt.subplots(2, 3, figsize=(16, 9))
#    for ax, noiseLevel, noiseLabels in zip(axObj.ravel(), clfNoiseLevel, clfNoiseLabel):
#        ax.scatter(featuresPCA[:, 0][noiseLabels == 1],
#                   featuresPCA[:, 1][noiseLabels == 1], color="b", marker=".",
#                   label="Normal", s=9)
##        ax.grid(True)
#        ax.tick_params(axis="y", labelsize=8)
#        ax.tick_params(axis="x", labelsize=8)
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_pca_features.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#        
#    # Colored by the ground truth
#    f, ax = plt.subplots(figsize=(8, 6))
#    uniqueLabels = -np.sort(-groundTruth["label"].unique())
#    for ind, label in enumerate(uniqueLabels):
#        sampleIndex = np.arange(0, len(groundTruth))
#        labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
#        
#        coords = featuresPCA[labeledSampleIndex, :]
#        if label != -1:
#            ax.scatter(coords[:, 0], coords[:, 1], s=9, 
#                       color='b', marker=".")
#        else:
#            ax.scatter(coords[:, 0], coords[:, 1], s=9, 
#                       color='r', marker="x", label="Class Abnormal")    
#        ax.tick_params(axis="y", labelsize=8)
#        ax.tick_params(axis="x", labelsize=8)
#    plt.legend(fontsize=8)
#
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_pca_2d_colored_by_truth.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#    
    '''
    Step 5: Date time based features visualizing.
    '''
#    featuresTmp = features[["length", "mean", "segMean_2", "dateTime"]].copy()
#    featuresTmp["day"] = featuresTmp["dateTime"].dt.day
#    featuresTmp["hour"] = featuresTmp["dateTime"].dt.hour
#    featuresTmp["month"] = featuresTmp["dateTime"].dt.month
#    
#    # Hour length plot
#    f, axObj = plt.subplots(2, 1, figsize=(10, 8))
#    ax = sns.boxplot(x="hour", y="length", data=featuresTmp, color="b", ax=axObj[0])
#    axObj[0].tick_params(axis="y", labelsize=10)
#    axObj[0].tick_params(axis="x", labelsize=10)
#    axObj[0].set_xlabel("hour", fontsize=10)
#    axObj[0].set_ylabel("length", fontsize=10)
#    
#    # Hour length plot
#    lengthHourTmp = featuresTmp.groupby(["hour"])["length"].agg(["mean", "std"]).reset_index()
#    ax = sns.catplot(x="hour", y="length", capsize=0.2, palette="YlGnBu_d",
#                     height=5, kind="point", data=featuresTmp, ax=axObj[1])
#    axObj[1].tick_params(axis="y", labelsize=10)
#    axObj[1].tick_params(axis="x", labelsize=10)
#    axObj[1].set_xlabel("hour", fontsize=10)
#    axObj[1].set_ylabel("length", fontsize=10)
#    plt.tight_layout()
#    
#    if plotSave:
#        f.savefig(".//Plots//0_EDA_catplot.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
#    
    '''
    Step 6: Find the abnormal seq in segment_1.
    '''
#    df = featureTmp[["segVar_1", "segMean_1", "segSlope_1"]].values
#    X_sc = StandardScaler()
#    df = X_sc.fit_transform(df)
#    
#    clf = DBSCAN(eps=0.3, min_samples=30)
#    labels = clf.fit_predict(df)
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure(figsize=(12, 9))
#    ax = Axes3D(fig)
#    ax.scatter(df[:, 0][labels != -1], df[:, 1][labels != -1],
#               df[:, 2][labels != -1], marker=".", color="k", s=30)
#    ax.scatter(df[:, 0][labels == -1], df[:, 1][labels == -1],
#               df[:, 2][labels == -1], marker="x", color="r", s=30)
#    
#    ax.tick_params(axis="y", labelsize=10)
#    ax.tick_params(axis="x", labelsize=10)
#    ax.tick_params(axis="z", labelsize=10)
#
#    ax.set_xlabel('segVar_1', fontdict={'size': 13}, color="g")
#    ax.set_ylabel('segMean_1', fontdict={'size': 13}, color="g")
#    ax.set_zlabel('segSlope_1', fontdict={'size': 13}, color="g")
#    plt.tight_layout()
#
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_seg_features_3d.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 6: KernelPCA for the original grouthtruth visvalizing.
    '''
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#
#    # pca ===>> 2 components 
#    components = 2
#    pca = PCA(n_components=components)
#    featuresPCA = pca.fit_transform(featureTmp.values.copy())
#    varRemain = pca.explained_variance_ratio_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # kpca ===>> 2 components
#    kpca = KernelPCA(n_components=components, kernel='rbf', gamma=0.0238)
#    featuresKPCA = kpca.fit_transform(featureTmp.values.copy())
#    varRemain = kpca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # Visualizing the ground truth of the data in feature space
#    fig, axObj = plt.subplots(1, 2, figsize=(13, 4))
#    uniqueTrueLabels = [1, 2, 3, -1]
#    
#    for ind, label in enumerate(uniqueTrueLabels):
#        sampleIndex = np.arange(0, len(groundTruth))
#        labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
#        
#        coordPCA = featuresPCA[labeledSampleIndex, :]
#        coordKPCA = featuresKPCA[labeledSampleIndex, :]
#        if label == -1:
#            axObj[0].scatter(coordPCA[:, 0], coordPCA[:, 1], s=9, alpha=0.4,
#                 color=colors[ind], marker="x", label="Class Abnormal")
#            axObj[1].scatter(coordKPCA[:, 0], coordKPCA[:, 1], s=9, alpha=0.4,
#                 color=colors[ind], marker="x", label="Class Abnormal")
#        else:
#            axObj[0].scatter(coordPCA[:, 0], coordPCA[:, 1], s=9, alpha=0.4,
#                 color=colors[ind], marker=markers[ind], label="Class " + str(int(label)))
#            axObj[1].scatter(coordKPCA[:, 0], coordKPCA[:, 1], s=9, alpha=0.4,
#                 color=colors[ind], marker=markers[ind], label="Class " + str(int(label)))
#        axObj[0].tick_params(axis="y", labelsize=8)
#        axObj[0].tick_params(axis="x", labelsize=8)
#        axObj[1].tick_params(axis="y", labelsize=8)
#        axObj[1].tick_params(axis="x", labelsize=8)
#        
#        axObj[0].legend(fontsize=7)
#        axObj[1].legend(fontsize=7)
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_ground_truth_pca_kpca_figure.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 7: OCSVM-features based anomaly detection.
    '''
#    # Standardizing the data
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#
#    # pca ===>> components
#    components = 15
#    pca = PCA(n_components=components)
#    featuresPCA = pca.fit_transform(featureTmp.values.copy())
#    varRemain = pca.explained_variance_ratio_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # kpca ===>> components
#    kpca = KernelPCA(n_components=components, kernel='rbf')
#    featuresKPCA = kpca.fit_transform(featureTmp.values.copy())
#    varRemain = kpca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#
#    # Detecting the anomaly data according to the features
#    test_param_nums = 10
#    params = {"nu":np.linspace(0.05, 0.95, test_param_nums),
#              "gamma":np.linspace(0.0667, 15, test_param_nums)}
#    pca_mat = np.zeros((test_param_nums, test_param_nums))
#    kpca_mat = np.zeros((test_param_nums, test_param_nums))
#    
#    y_true = np.where(groundTruth["label"] != -1, 1, -1)
#    for i, item_nu in enumerate(params["nu"]):
#        for j, item_gamma in enumerate(params["gamma"]):
#            clf = OneClassSVM(kernel='rbf', nu=item_nu, gamma=item_gamma, verbose=True)
#            clf.fit(featuresPCA)
#            noiseLevel_pca = clf.decision_function(featuresPCA) - clf.intercept_
#            fpr_pca, tpr_pca, _ = roc_curve(y_true, noiseLevel_pca)
#            pca_mat[i, j] = auc(fpr_pca, tpr_pca)
#
#            clf = OneClassSVM(kernel='rbf', nu=item_nu, gamma=item_gamma, verbose=True)
#            clf.fit(featuresKPCA)
#            noiseLevel_kpca = clf.decision_function(featuresKPCA) - clf.intercept_
#            fpr_kpca, tpr_kpca, _ = roc_curve(y_true, noiseLevel_kpca)
#            kpca_mat[i, j] = auc(fpr_kpca, tpr_kpca)
#    
#    fig, axObj = plt.subplots(1, 2, figsize=(10, 4))
#    for ind, (ax, featureCorr) in enumerate(zip(axObj, [pca_mat, kpca_mat])):
#        ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
#                         cmap="Blues", fmt='.3f', annot=True,
#                         annot_kws={'size':5.5,'weight':'bold'}, ax=ax)
#        ax.tick_params(axis="y", labelsize=7, rotation=0)
#        ax.tick_params(axis="x", labelsize=7)
#        
#        ax.set_xlabel("nu", fontsize=8)
#        ax.set_ylabel("gamma", fontsize=8)
#        ax.set_xticklabels([round(i, 2) for i in params["nu"]])
#        ax.set_yticklabels([round(i, 2) for i in params["gamma"]])
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=7)
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_ocsvm_detection_heatmap.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")

    '''
    Step 8: LOF-features based anomaly detection.
    '''
#    # Standardizing the data
#    featureTmp = features.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#
#    # pca ===>> components
#    components = 15
#    pca = PCA(n_components=components)
#    featuresPCA = pca.fit_transform(featureTmp.values.copy())
#    varRemain = pca.explained_variance_ratio_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # kpca ===>> components
#    kpca = KernelPCA(n_components=components, kernel='rbf')
#    featuresKPCA = kpca.fit_transform(featureTmp.values.copy())
#    varRemain = kpca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#
#    # Detecting the anomaly data according to the features
#    test_param_nums = 15
#    params = {"nn":np.sort(np.random.choice([i for i in range(2, 700)], test_param_nums, replace=False))}
#    pca_mat = np.zeros((test_param_nums, 1))
#    kpca_mat = np.zeros((test_param_nums, 1))
#    original_mat = np.zeros((test_param_nums, 1))
#    
#    y_true = np.where(groundTruth["label"] != -1, 1, -1)
#    for i, item_nn in enumerate(params["nn"]):
#        # pca
#        clf = LocalOutlierFactor(n_neighbors=item_nn)
#        clf.fit(featuresPCA)
#        
#        noiseLevel_pca = clf.negative_outlier_factor_
#        fpr_pca, tpr_pca, _ = roc_curve(y_true, noiseLevel_pca)
#        pca_mat[i] = auc(fpr_pca, tpr_pca)
#        
#        # kernel pca
#        clf = LocalOutlierFactor(n_neighbors=item_nn)
#        clf.fit(featuresKPCA)
#        
#        noiseLevel_kpca = clf.negative_outlier_factor_
#        fpr_kpca, tpr_kpca, _ = roc_curve(y_true, noiseLevel_kpca)
#        kpca_mat[i] = auc(fpr_kpca, tpr_kpca)        
#        
#        # original
#        clf = LocalOutlierFactor(n_neighbors=item_nn)
#        clf.fit(featureTmp)
#        
#        noiseLevel_original = clf.negative_outlier_factor_
#        fpr_original, tpr_original, _ = roc_curve(y_true, noiseLevel_original)
#        original_mat[i] = auc(fpr_original, tpr_original)
#        
#    fig, ax = plt.subplots(figsize=(13, 4))
#    plt.plot(params["nn"], pca_mat, color=colors[0], lw=1.3,
#             marker="o", linestyle="--",
#             markersize=4, label="PCA LOF")
#    plt.plot(params["nn"], kpca_mat, color=colors[1], lw=1.3,
#             marker="s",linestyle="-.",
#             markersize=4, label="KPCA LOF")
#    plt.plot(params["nn"], original_mat, color=colors[2], lw=1.3,
#             marker="^", linestyle="-",
#             markersize=4, label="Original LOF")
#    plt.xlim(params["nn"].min()-5, params["nn"].max()+5)
#    ax.tick_params(axis="both", labelsize=8)
#    plt.legend(fontsize=8)
#    plt.xlabel("Number of neighbors", fontsize=8)
#    plt.ylabel("AUC", fontsize=8)
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_lof_detection_heatmap.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    
    '''
    Step 9: Feature-based clustering results.(NOT READY)
    '''
#    featureNormal = features.loc[groundTruth["ind"][groundTruth["label"] != -1].values.tolist()]
#    featureNormal.reset_index(drop=True)
#    y_true = groundTruth["label"][groundTruth["label"] != -1].values.tolist()
#
#    featureTmp = featureNormal.copy()
#    featureTmp.drop(["dateTime", "no"], axis=1, inplace=True)
#    X_sc = StandardScaler()
#    for name in featureTmp.columns:
#        featureTmp[name].fillna(featureTmp[name].mean(), inplace=True)
#        featureTmp[name] = X_sc.fit_transform(featureTmp[name].values.reshape(len(featureTmp), 1))
#    
#    # pca ===>> components
#    components = 15
#    pca = PCA(n_components=components)
#    featuresPCA = pca.fit_transform(featureTmp.values.copy())
#    varRemain = pca.explained_variance_ratio_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    # kpca ===>> components
#    kpca = KernelPCA(n_components=components, kernel='rbf')
#    featuresKPCA = kpca.fit_transform(featureTmp.values.copy())
#    varRemain = kpca.lambdas_
#    print("\n@Number of features {}, after compression {}.".format(features.shape[1], components))
#    print("@Information reamins : {}".format(sum(varRemain)))
#    
#    dist_mat_pca = pairwise_distances(featuresPCA, metric="euclidean")
#    dist_mat_kpca = pairwise_distances(featuresKPCA, metric="euclidean")
#    
#    # Clustering according to the features
#    test_param_nums = 8
#    params = {"clusters":np.sort(np.random.choice([i for i in range(2, 15)], test_param_nums, replace=False)),
#              "gamma": np.linspace(0.001, 15, test_param_nums)}  
#    pca_res_mat_sc = np.zeros((test_param_nums, test_param_nums))
#    pca_res_mat_ap = np.zeros((test_param_nums, test_param_nums))
#    
#    kpca_res_mat_sc = np.zeros((test_param_nums, test_param_nums))
#    kpca_res_mat_ap = np.zeros((test_param_nums, test_param_nums))
#    
#    kmeans_baseline = np.zeros((test_param_nums, test_param_nums))
#    
#    for i, item_c in enumerate(params["clusters"]):
#        for j, item_gamma in enumerate(params["gamma"]):
#            print(i, j)
#            dist_mat_pca_gamma = np.exp( np.divide(-np.square(dist_mat_pca), 2 * item_gamma ** 2) )
#            dist_mat_kpca_gamma = np.exp( np.divide(-np.square(dist_mat_kpca), 2 * item_gamma ** 2) )
#            
##            clf = SpectralClustering(n_clusters=item_c, affinity='precomputed',
##                                     n_jobs=-1, eigen_solver='arpack')
##            labels = clf.fit_predict(dist_mat_pca_gamma)
##            pca_res_mat_sc[i, j] = v_measure_score(y_true, labels)
##
##            labels = clf.fit_predict(dist_mat_kpca_gamma)
##            kpca_res_mat_sc[i, j] = v_measure_score(y_true, labels)
#            
#            clf = KMeans(n_clusters=item_c)
#            labels = clf.fit_predict(featureTmp)
#            kmeans_baseline[i, j] = v_measure_score(y_true, labels)
#            
#            clf = AgglomerativeClustering(n_clusters=item_c,
#                                          affinity='precomputed',
#                                          linkage="average")
#            labels = clf.fit_predict(dist_mat_pca_gamma)
#            pca_res_mat_ap[i, j] = v_measure_score(y_true, labels)
#
#            labels = clf.fit_predict(dist_mat_kpca_gamma)
#            kpca_res_mat_ap[i, j] = v_measure_score(y_true, labels)
#
#    fig, axObj = plt.subplots(2, 2, figsize=(10, 7))
#    for ind, (ax, featureCorr) in enumerate(zip(axObj.ravel(), [pca_res_mat_sc, kpca_res_mat_sc, pca_res_mat_ap, kpca_res_mat_ap])):
#        ax = sns.heatmap(featureCorr, xticklabels=1, yticklabels=1,
#                         cmap="Blues", fmt='.2f', annot=True,
#                         annot_kws={'size':5.5,'weight':'bold'}, ax=ax)
#        ax.tick_params(axis="y", labelsize=7, rotation=0)
#        ax.tick_params(axis="x", labelsize=7)
#        
#        ax.set_xlabel("clusters", fontsize=8)
#        ax.set_ylabel("gamma", fontsize=8)
#        ax.set_xticklabels([round(i, 2) for i in params["clusters"]])
#        ax.set_yticklabels([round(i, 2) for i in params["gamma"]])
#        cbar = ax.collections[0].colorbar
#        cbar.ax.tick_params(labelsize=7)    
#    plt.tight_layout()
#    
#    if plotSave:
#        plt.savefig(".//Plots//0_EDA_feature_based_clustering_heatmap.pdf", dpi=500, bbox_inches="tight")
#        plt.close("all")
#    if not plotShow:
#        plt.close("all")
    '''
    Step 10: Plot normal signals with abnormal signals.
    '''
    noiseInd = [i for i in list(features.index) if groundTruth["label"].loc[i] == -1]
    noiseInd = np.random.choice(noiseInd, size=10, replace=False)
    noiseTs = [ts[i] for i in noiseInd]
    
    normalInd = [i for i in list(features.index) if groundTruth["label"].loc[i] != -1]
    normalInd = np.random.choice(normalInd, size=300, replace=False)
    normalTs = [ts[i] for i in normalInd]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    for signal in noiseTs:
        ax.plot(signal, color="r", linewidth=1.5)
    for signal in normalTs:
        ax.plot(signal, color="k", linewidth=1.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlim(0, 157)
    ax.set_ylim(0, 7.5)

    if plotSave:
        plt.savefig(".//Plots//0_EDA_anomaly_normal_mixed.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
