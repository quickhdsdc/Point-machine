#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 00:40:52 2019

@author: michaelyin1994
"""

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import seaborn as sns
import time
import pickle
from functools import wraps
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from StackedDenoisingAE import StackedDenoisingAE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def save_data(data, fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Save file to {}".format(fileName))
    f = open(fileName, 'wb')
    pickle.dump(data, f)
    f.close()
    print("Save successed !")
    print("-------------------------------------")


def load_data(fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Load file from {}".format(fileName))
    f = open(fileName, 'rb')
    data = pickle.load(f)
    f.close()
    print("Load successed !")
    print("-------------------------------------")
    return data


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fn.__name__ + " took {:.5f}".format(end-start) + " seconds")
        return result
    return measure_time


@timefn
def replace_inf_with_nan(data):
    featureNameList = list(data.columns)
    for name in featureNameList:
        data[name].replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


@timefn
def basic_feature_report(data):
    basicReport = data.isnull().sum()
    sampleNums = len(data)
    basicReport = pd.DataFrame(basicReport, columns=["missingNums"])
    basicReport["missingPrecent"] = basicReport["missingNums"]/sampleNums
    basicReport["nuniqueValues"] = data.nunique(dropna=False).values
    basicReport["types"] = data.dtypes.values
    basicReport.reset_index(inplace=True)
    basicReport.rename(columns={"index":"featureName"}, inplace=True)
    dataDescribe = data.describe([0.01, 0.5, 0.99, 0.995, 0.9995]).transpose()
    dataDescribe.reset_index(inplace=True)
    dataDescribe.rename(columns={"index":"featureName"}, inplace=True)
    basicReport = pd.merge(basicReport, dataDescribe, on='featureName', how='left')
    return basicReport


def drop_most_empty_features(data=None, precent=None):
    assert precent, "@MichaelYin: Invalid missing precent !"
    dataReport = basic_feature_report(data)
    featureName = list(dataReport["featureName"][dataReport["missingPrecent"] >= precent].values)
    data.drop(featureName, axis=1, inplace=True)
    return data, featureName


def plot_single_record(data=[], record=0):
    if len(data) == 0:
        return None

    # Indexing the data
    tmp = data.query("record == " + str(record))

    # Plot the data
    f, ax = plt.subplots()
    plt.plot(tmp.query("phase == " + str(0))["current"].values[0], color="red", linewidth=2, linestyle="-")
    plt.plot(tmp.query("phase == " + str(1))["current"].values[0], color="blue", linewidth=2, linestyle="-")
    plt.plot(tmp.query("phase == " + str(2))["current"].values[0], color="green", linewidth=2, linestyle="-")
    plt.plot(tmp.query("phase == " + str(3))["current"].values[0], color="black", linewidth=2, linestyle="-")

    plt.legend(["Phase_1", "Phase_2", "Phase_3", "Power"])
    plt.xlim(0, )
    plt.title("Four series of record {}".format(record))
    
###############################################################################
###############################################################################
class LoadSave(object):
    def __init__(self, fileName=None):
        self._fileName = fileName
    
    def save_data(self, data=None, path=None):
        if path is None:
            assert self._fileName != None, "Invaild file path !"
            self.__save_data(data)
        else:
            self._fileName = path
            self.__save_data(data)
    
    def load_data(self, path=None):
        if path is None:
            assert self._fileName != None, "Invaild file path !"
            return self.__load_data()
        else:
            self._fileName = path    
            return self.__load_data()
        
    def __save_data(self, data=None):
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._fileName))
        f = open(self._fileName, "wb")
        pickle.dump(data, f)
        f.close()
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")
        
    def __load_data(self):
        assert self._fileName != None, "Invaild file path !"
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._fileName))
        f = open(self._fileName, 'rb')
        data = pickle.load(f)
        f.close()
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data
###############################################################################
###############################################################################
def generating_cutted_sequences(data=[], window=20, stride=10,
                                portion=0.4, verboseRound=500):
    '''
    data: list-like
        Each element in the data is a list which contains points of sequences.
    '''
    if len(data) == 0:
        return None
    
    print("\n@Michael Yin: Start cutting sequences")
    print("=====================================================")
    print("Start time : {}".format(datetime.now()))
    cuttedSeqs = []
    for ind, item in enumerate(data):
        if ind % verboseRound == 0:
            print("    --Current cutting at the {}, total is {}.".format(ind, len(data)))
        cuttedSeqs.append(sequence_to_subsequence(item, window, stride, portion))
    print("End time : {}".format(datetime.now()))
    print("=====================================================")
    return cuttedSeqs


def generating_equal_sequences(data=[], padding_val=0,
                               mode="max_length", length=None):
    '''
    data: list-like
        Each element in the data is a list which contains points of sequences.
    '''
    if len(data) == 0:
        return None
    
    if not isinstance(data, list):
        raise TypeError("Invalid data type !")
    
    if mode not in ["max_length", "min_length", "pre_defined", "mean_length"]:
        raise TypeError("Invalid mode !")
    
    if mode == "pre_defined" and length is None:
        raise TypeError("Invalid length param !")
        
    if mode == "max_length":
        length = max([len(i) for i in data])
    elif mode == "min_length":
        length = min([len(i) for i in data])
    elif mode == "mean_length":
        length = int(round(sum([len(i) for i in data]) / len(data)))
    
    print("\n@Michael Yin: Start padding sequences")
    print("=====================================================")
    print("@Start time : {}".format(datetime.now()))
    paddingRes, lengthDiffSum, processNumsSum = [], [], 0
    for i in tqdm(range(len(data))):
        seq = data[i]
        segRes, lengthDiff, processNums = sequence_to_padding_sequence(seq, padding_val=padding_val, length=length)
        
        paddingRes.append(segRes)
        lengthDiffSum.append(lengthDiff)
        processNumsSum += processNums
    print("@End time : {}".format(datetime.now()))
    if processNumsSum:
        print("@Process nums {}, average process pts {:.3f}, length is {}.".format(processNumsSum,
              sum(lengthDiffSum)/processNumsSum, length))
    print("=====================================================")
    return paddingRes


def sequence_to_subsequence(seq=[], window=20, stride=10, portion=0.4):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/08/05
    Modified: 2019/08/05
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Cutting the sequence accroding to the size of the window and stride
    
    @Parameters:
    ----------
    seq: list-like or numpy-array-like
        The path of the database.
    
    window: pandas DataFrame-like
        How many data points in a window.
    
    stride: int-like
        The number of steps from the start point of a cutted window to another
        window.
        
    @Return:
    ----------
    segRes: dict-like,
            cuttting results. It acts like: {0: [[cutted seq], [mean], [var]], ...}
    '''
    if len(seq) <= window:
        return {0:[[seq], [np.mean(seq)], [np.var(seq)]]}
    if portion > 1 or portion < 0:
        return {}
    
    # Step 1: generating the segment points.
    segmentPoints, count = [], 0
    while(len(seq) - count >= window):
        segmentPoints.append(count)
        count += stride
    
    # Step 2: Checking the number of points that not included in the segments.
    remainPts = len(seq) - segmentPoints[-1] - window
    if remainPts / window >= portion:
        segmentPoints.append(len(seq) - window)
    
    # Step 3: Start cutting the current.
    segRes = {}
    for ind, item in enumerate(segmentPoints):
        tmp = seq[item:(item+window)]
        segRes[ind] = [tmp, np.mean(tmp), np.var(tmp), remainPts/len(seq)]
    return segRes


def sequence_to_padding_sequence(seq=[], padding_val=0, length=None):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/09/13
    Modified: 2019/09/13
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    padding the sequence to the equal length.
    
    @Parameters:
    ----------
    seq: list-like
        A sequence need to be padded.
    
    padding_val: float-like
        Value that used to padding.
    
    length: int-like
        Show the process.
        
    @Return:
    ----------
    paddingRes: list-like,
            Padding results: [paddingSea, mean(paddingSea), var(paddingSea), None]
    '''
    if len(seq) > length:
        tmp = seq[:length].copy()
        lengthDiff = (len(seq) - length)
        processNums = 1
        
    elif len(seq) < length:
        needToPadPtsNum = length - len(seq)
        tmp = seq + [padding_val] * needToPadPtsNum
        lengthDiff = (length - len(seq))
        processNums = 1
        
    else:
        tmp = seq.copy()
        lengthDiff = 0
        processNums = 0
    return {0:[tmp, np.mean(tmp), np.var(tmp), None]}, lengthDiff, processNums

###############################################################################
###############################################################################
class LoadSaveFromDatabase(object):
    '''
    ----------
    Author: Michael Yin
    Date: 2018/12/26
    Modified: 2019/07/12
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Loading and saving data in the SQL format.
    
    @Parameters:
    ----------
    databsaePath: str-like
        The path of the database.
    
    sheetName: pandas DataFrame-like
        The dataframe that need to be saved.
        
    @Return:
    ----------
    None
    '''
    def __init__(self, databasePath=None, sheetName=None):
        self.databasePath = databasePath
        self.sheetName = sheetName
    
    @timefn
    def load_data(self, databasePath=None, sheetName=None):
        if databasePath is None:
            databasePath = self.databasePath
        if sheetName is None:
            sheetName = self.sheetName
        
        assert databasePath, "Invalid data base path !"
        assert sheetName, "Invalid path or sheet name!"
        
        print("\n--------------------------------------------------")
        print("LOADING FROM DATABASE PATH: {}.".format(databasePath))
        print("SHEET NAME :{}.".format(sheetName))
        sqlConnection = sqlite3.connect(databasePath)
        data = pd.read_sql_query("SELECT * FROM " + sheetName, sqlConnection)
        print("\nSUCCESSED !")
        print("----------------------------------------------------")
        sqlConnection.commit()
        sqlConnection.close()
        return data
    
    @timefn
    def save_data(self, data, databasePath=None, sheetName=None, if_exists='replace'):
        if databasePath is None:
            databasePath = self.databasePath
        if sheetName is None:
            sheetName = self.sheetName
        
        assert databasePath, "Invalid data base path !"
        assert sheetName, "Invalid path or sheet name!"
        
        print("\n--------------------------------------------------")
        print("SAVING TO DATABASE PATH: {}.".format(databasePath))
        print("SHEET NAME :{}.".format(sheetName))
        sqlConnection = sqlite3.connect(databasePath)
        data.to_sql(sheetName, sqlConnection, if_exists=if_exists, index=False)
        print("\nSUCCESSED !")
        print("----------------------------------------------------")
        sqlConnection.commit()
        sqlConnection.close()

###############################################################################
###############################################################################
class ReduceMemoryUsage():
    '''
    ----------
    Author: Michael Yin
    Date: 2018/12/26
    Modified: 2019/07/12
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Reduce the pandas dataframe memory usage.
    
    @Parameters:
    ----------
    data: pandas DataFrame-like
        The dataframe that need to be reduced memory usage.
    
    verbose: bool
        Whether to print the memory reducing information or not.
        
    @Return:
    ----------
    Memory-reduced dataframe.
    '''
    def __init__(self, data=None, verbose=True):
        self._data = data
        self._verbose = verbose
    
    def types_report(self, data):
        dataTypes = list(map(str, data.dtypes.values))
        basicReport = pd.DataFrame(dataTypes, columns=["types"])
        basicReport["featureName"] = list(data.columns)
        return basicReport
    
    @timefn
    def reduce_memory_usage(self):
        self.__reduce_memory()
        return self._data
    
    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memoryStart = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage of data is {:5f} MB.".format(memoryStart))
        self._types = self.types_report(self._data)
        for ind, name in enumerate(self._types["featureName"].values):
            # WARNING: Unstable part.
            featureType = str(self._types[self._types["featureName"] == name]["types"].iloc[0])
            
            if (featureType != "object") and (featureType != "datetime64[ns]"):
                # Print the error information
                try:
                    featureMin = self._data[name].min()
                    featureMax = self._data[name].max()
                    if "int" in featureType:
                        # np.iinfo for reference:
                        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                        
                        # numpy data types reference:
                        # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                        if featureMin > np.iinfo(np.int8).min and featureMax < np.iinfo(np.int8).max:
                            self._data[name] = self._data[name].astype(np.int8)
                        elif featureMin > np.iinfo(np.int16).min and featureMax < np.iinfo(np.int16).max:
                            self._data[name] = self._data[name].astype(np.int16)
                        elif featureMin > np.iinfo(np.int32).min and featureMax < np.iinfo(np.int32).max:
                            self._data[name] = self._data[name].astype(np.int32)
                        elif featureMin > np.iinfo(np.int64).min and featureMax < np.iinfo(np.int64).max:
                            self._data[name] = self._data[name].astype(np.int64)
                    else:
                        if featureMin > np.finfo(np.float32).min and featureMax < np.finfo(np.float32).max:
                            self._data[name] = self._data[name].astype(np.float32)
                        else:
                            self._data[name] = self._data[name].astype(np.float64)
                    if featureMin > np.finfo(np.float32).min and featureMax < np.finfo(np.float32).max:
                        self._data[name] = self._data[name].astype(np.float32)
                    else:
                        self._data[name] = self._data[name].astype(np.float64)
                except Exception as error:
                    print("\n--------ERROR INFORMATION---------")
                    print(error)
                    print("Error on the {}".format(name))
                    print("--------ERROR INFORMATION---------\n")
            if self._verbose == True:
                print("Processed {} feature({}), total is {}.".format(ind+1, name, len(self._types)))
        memoryEnd = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage after optimization: {:.5f} MB.".format(memoryEnd))
            print("@Decreased by {}%.".format(100 * (memoryStart - memoryEnd) / memoryStart))
        print("-------------------------------------------")

###############################################################################
###############################################################################
def plot_roc_auc_curve(y_test=None, y_pred=None, name="dense"):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/08/13
    Modified: 2019/08/13
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Plot the ROC curve, and calculate the micro AUC of ROC.
    
    @Parameters:
    ----------
    y_test, y_pred: numpy array-like
        Real labels and its predictions. For binary classifiction problem, it is 
        a 2-D matrix. For N-class problem, it is a N-D matrix.
    
    name: str-like
        The figure name for the ROC curve.

    @Return:
    ----------
    AUC for each class.
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot roc_auc curve
    plt.figure(figsize=(16, 9))
    lw = 3

    # @BugToBeFixed: The list of colors only contain 10 colors.
    #                Need to use cycle to generate iterator.
    colors = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    for i, color in zip(range(y_test.shape[1]), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.5f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Micro Roc: {:0.5f}'.format(roc_auc["micro"]))
    plt.legend(loc="lower right")
    plt.savefig(".//Plots//" + name + ".png", bbox_inches="tight", dpi=500)
    return roc_auc


def plot_feature_importance(importances=[], topK=10):
    if len(importances) == 0:
        return None
    
    if len(importances) < topK:
        topK = len(importances)
    
    imp_cols = [name for name in importances.columns if "importances" in name]
    mean_importance = np.mean(importances[imp_cols].values, axis=1)
    
    df = pd.DataFrame(None)
    df["featureName"] = ["col_" + str(i) for i in importances["featureName"].values]
    df["importances"] = mean_importance
    df.sort_values(by="importances", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax = sns.barplot(x="featureName",
                     y="importances",
                     data=df[:topK],
                     palette="Blues_d",
                     ax=ax, order=None)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("feature name", fontsize=10)
    ax.set_ylabel("importances", fontsize=10)
###############################################################################
###############################################################################
def rf_clf_training(train, test, rfParams=None, numFolds=3, stratified=True,
                    shuffle=True, randomState=2019):
    # Specify the cross validated method
    if stratified == True:
        folds = StratifiedKFold(n_splits=numFolds, shuffle=shuffle,
                                random_state=randomState)
    else:
        folds = KFold(n_splits=numFolds, shuffle=shuffle, random_state=randomState)
    
    # Make oof predictions, test predictions
    importances = pd.DataFrame(None)
    importances["featureName"] = list(train.drop(["target"], axis=1).columns)
    score = np.zeros((numFolds, 4))
    
    oof_pred = np.zeros((len(train), 1))
    y_pred = np.zeros((len(test), ))
    
    # Start training
    dataTrain, dataTrainLabel = train.drop("target", axis=1), train["target"]
    ###########################################################################
    for fold, (trainId, validationId) in enumerate(folds.split(dataTrain, dataTrainLabel)):
        # iloc the split data
        X_train, y_train = dataTrain.iloc[trainId].values, dataTrainLabel.iloc[trainId].values
        X_valid, y_valid = dataTrain.iloc[validationId].values, dataTrainLabel.iloc[validationId].values
        
        # Start training
        clf = RandomForestClassifier(**rfParams)
        clf.fit(X_train, y_train)
        
        # Some of scores
        importances["importances_" + str(fold)] = clf.feature_importances_
        score[fold, 0] = fold
        score[fold, 1] = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        score[fold, 2] = roc_auc_score(y_valid, clf.predict_proba(X_valid)[:, 1])
        score[fold, 3] = clf.oob_score_
        
        # Get the oof_prediction and the test prediction results
        oof_pred[validationId, 0] = clf.predict_proba(X_valid)[:, 1]
        y_pred += clf.predict_proba(test.values)[:, 1] / numFolds
                                    
    score = pd.DataFrame(score, columns=["fold", "trainScore", "validScore", "oofScores"])
    print("\n@Average score:")
    print("==================================================================")
    print("    ---Average train score {:.7f}.".format(score["trainScore"].mean()))
    print("    ---Average valid score {:.7f}.".format(score["validScore"].mean()))
    print("    ---CV score: {:.7f}.".format(roc_auc_score(train['target'].values, oof_pred)))
    print("==================================================================")
    
    return score, importances, y_pred


###############################################################################
###############################################################################
def weighting_rep(df=[], norm_factor=0.05, bin_name="bin_freq_10"):
    if len(df) == 0:
        return None
    if bin_name not in df.columns:
        raise TypeError("Bin name not in the df !")
    
    # Preparing the reweighting
    repTs = df[[name for name in df.columns if "rep" in name]]
    segRes = df[[name for name in df.columns if "rep" not in name]]
    
    # Weights function
    #----------------------------------------------------
    segRes["weights"] = 2 / (1 + np.exp(norm_factor * segRes[bin_name].values))
    #----------------------------------------------------
    
    weightsNorm = segRes.groupby(["ind"])["weights"].sum().reset_index().rename({"weights":"norm_factor"}, axis=1)
    segRes = pd.merge(segRes, weightsNorm, how="left", on="ind")
    segRes["weights_norm"] = segRes["weights"] / segRes["norm_factor"]
    
    # Reweighting
    signalRep_weight = repTs.multiply(segRes["weights_norm"].values, axis=0).copy()
    signalRep_weight["flag"] = segRes["ind"]
    signalRep_weight = signalRep_weight.groupby(["flag"]).mean().values
    
    # Non-weighting
    repTs["ind"] = segRes["ind"].values
    signalRep_mean = repTs.groupby(["ind"]).mean().values
    
    return signalRep_mean, signalRep_weight