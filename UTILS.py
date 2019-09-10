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

from StackedDenoisingAE import StackedDenoisingAE
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb

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
        print("@timefn: " + fn.__name__ + " took {:5f}".format(end-start) + " seconds")
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
def generating_cutted_sequences(data=[], window=20, stride=10, portion=0.4, verboseRound=500):
    '''
    data: list-like
        Each element in the data is a list which contains points of sequences.
    '''
    if len(data) == 0:
        return None
    
    print("@Michael Yin: Start cutting sequences")
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
            print("@Memory usage after optimization: {:5f} MB.".format(memoryEnd))
            print("@Decreased by {}%".format(100 * (memoryStart - memoryEnd) / memoryStart))
        print("-------------------------------------------")

###############################################################################
###############################################################################
def lgb_clf_training(train, test, lgbParams=None, numFolds=3, stratified=True,
                     shuffle=True, randomState=2812):
    # Specify the cross validated method
    if stratified == True:
        folds = StratifiedKFold(n_splits=numFolds, shuffle=shuffle,
                                random_state=randomState)
    else:
        folds = KFold(n_splits=numFolds, shuffle=shuffle, random_state=randomState)
    
    # Start training
    dataTrain = train.drop("target", axis=1)
    dataTrainLabel = train["target"]
    importances = pd.DataFrame(None)
    importances["featureName"] = list(train.drop(["target"], axis=1).columns)
    score = np.zeros((numFolds, 4))
    
    ###########################################################################
    for fold, (trainId, validationId) in enumerate(folds.split(dataTrain, dataTrainLabel)):
        # iloc the split data
        X_train, y_train = dataTrain.iloc[trainId], dataTrainLabel.iloc[trainId]
        X_valid, y_valid = dataTrain.iloc[validationId], dataTrainLabel.iloc[validationId]
        
        # Start training
        clf = lgb.LGBMClassifier(**lgbParams)
        clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values),
                                                          (X_valid.values, y_valid.values)],
                                                          early_stopping_rounds=20, eval_metric="auc")
        importances["importances_" + str(fold)] = clf.feature_importances_
        scoreTmp = clf.evals_result_
        scoreKeys = list(scoreTmp.keys())
        score[fold, 0] = fold
        score[fold, 1] = scoreTmp[scoreKeys[0]]["auc"][clf.best_iteration_ - 1]
        score[fold, 2] = scoreTmp[scoreKeys[1]]["auc"][clf.best_iteration_ - 1]
        score[fold, 3] = clf.best_iteration_
    score = pd.DataFrame(score, columns=["fold", "trainScore", "validScore", "bestIters"])
    print("\n@Average score:")
    print("==================================================================")
    print("    ---Average train score {}".format(score["trainScore"].mean()))
    print("    ---Average valid score {}".format(score["validScore"].mean()))
    print("==================================================================")
    
    print("\n@Start the finall fit:")
    print("==================================================================")
    print("@Start time : {}".format(datetime.now()))
    lgbParams["n_estimators"] = int(score["bestIters"].max())
    clf = lgb.LGBMClassifier(**lgbParams)
    clf.fit(dataTrain.values, dataTrainLabel.values)
    importances["importances_finall"] = clf.feature_importances_
    y_pred = clf.predict_proba(test.values)[:, 1]
    print("@End time : {}".format(datetime.now()))
    print("==================================================================")
    
    return score, importances, y_pred


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
    
    imp = importances.mean(axis=1).reset_index().rename({"index":"featureName",
                          0:"importances"}, axis=1)
    imp.sort_values(by="importances", inplace=True, ascending=False)
    
    plt.figure(figsize=(16, 9))
    sns.barplot(x=imp["featureName"][:topK].apply(str).values, y=imp["importances"][:topK].values, palette="Blues")
    
###############################################################################
###############################################################################
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, auc
def simple_lgb_clf(X_train, X_valid, X_test, y_train, y_valid, y_test, name="dense"):
    '''
    X_train, X_valid, X_test: numpy-array like.
    y_train, y_valid, y_test: numpy-array like.
    name: the roc curve picture name.
    '''
    lgbParams = {'n_estimators': 5000,
                 'objective': 'multiclass',
                 'boosting_type': 'gbdt',
                 'n_jobs': -1,
                 'learning_rate': 0.05,
                 'num_leaves': 40, # important
                 'max_depth': 8, # important
                 # 'min_split_gain': 0,
                 # 'min_child_samples': 20,
                 'subsample': 0.8,
                 'subsample_freq': 1,
                 'colsample_bytree':0.8,
                 'reg_alpha': 0.56154,
                 'reg_lambda': 0.5735294,
                 'silent': 1}
    clf = lgb.LGBMClassifier(**lgbParams)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, eval_metric="softmax")
    y_pred = clf.predict_proba(X_test)
    Y_test = to_categorical(y_test, len(np.unique(y_train)))
    auc = roc_auc_curve(Y_test, y_pred, name=name)
    return auc

###############################################################################
###############################################################################
def get_dae_rep(trainData=None, vaildData=None, segRes=None,
                windowSize=None, stride=None,
                weight_regularizer=[], activity_regularizer=[],
                n_hid=[20], dropout=[0.02], enc_act=["relu"], nb_epoch=80,
                batch_size=256, early_stop_rounds=15, 
                path=".//Data//rep_data//", fileName="rep_0", save_rep=False):
    
    # Check the parameters
    if windowSize is None or stride == None:
        return

    sdae = StackedDenoisingAE(n_layers=len(n_hid), n_hid=n_hid, dropout=dropout,
                              weight_regularizer=weight_regularizer,
                              activity_regularizer=activity_regularizer,
                              nb_epoch=nb_epoch, enc_act=["relu"],
                              early_stop_rounds=early_stop_rounds, dec_act=["linear"],
                              batch_size=batch_size, bias=True)
    model, (dense_train, dense_val, dense_test), recon_mse = sdae.get_pretrained_sda(trainData, vaildData,
           vaildData, dir_out='.//Models//', write_model=False)

    # Combining the subsequences into a single sequence
    newData = np.concatenate([dense_train, dense_test], axis=0)
    newData = pd.DataFrame(newData)
    newData["flag"] = segRes["ind"]
    signalRep = newData.groupby(["flag"]).mean().values

    # Save the data
    if save_rep:
        ls = LoadSave(path + fileName + "_" + str(windowSize) + "_"
                      + str(stride) + ".pkl")
        ls.save_data(signalRep)
    return signalRep


