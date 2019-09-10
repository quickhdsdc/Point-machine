#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:33:20 2019

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
import fastdtw
from numba import jit
from datetime import datetime

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
@jit
def dynamic_time_warping(X, Y):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/07/13
    Modified: 2019/07/13
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Version 1:
        Calculating the standard dynamic time warping distance.
    
    @Parameters:
    ----------
    X: list-like or numpy array-like
        Time series X.
    
    Y: list-like or numpy array-like
        Time series Y.
        
    @Return:
    ----------
    dtw distance between series X and Y.
    '''
    length_X, length_Y = len(X), len(Y)
    
    # Initializing some parameters
    dp = np.zeros((length_X + 1, length_Y + 1))
    dp[0, 1:] = np.inf
    dp[1:, 0] = np.inf
    dpPanel = dp[1:, 1:]
    
    # Initializing the distance matrix
    for i in range(length_X):
        for j in range(length_Y):
            dpPanel[i, j] = np.abs((X[i] - Y[j]))
    
    # Calculation of the dp matrix
    for i in range(length_X):
        for j in range(length_Y):
            dpPanel[i, j] += min(dp[i+1, j], dp[i, j+1], dp[i, j])
    return dp[-1, -1]


@jit
def longest_common_subsequence(X, Y, minDistCond=50, minPts=0.2):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/07/13
    Modified: 2019/07/13
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Calculating the longest common susequences.
    
    @Parameters:
    ----------
    X: list-like or numpy array-like
        Time series X.
    
    Y: list-like or numpy array-like
        Time series Y.
    
    minDistCond, minPts: float
        dist <= minDistCond and pts <= minPts, then 2 points can
        be viewed as the same.
    
    @Return:
    ----------
    lcss distance between series X and Y.
    '''
    length_X, length_Y = len(X), len(Y)
    
    # Initializing the dp matrix
    dp = np.zeros((length_X + 1, length_Y + 1))
    
    # Fill the elements according to the dp rule
    for i in range(1, length_X + 1):
        for j in range(1, length_Y + 1):
            pointDist = np.abs(X[i-1] - Y[j-1])
            if pointDist <= minDistCond and np.abs(i - j) <= minPts:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    return dp[-1, -1]


def frechet_distance(X, Y):
    pass


@jit
def edit_distance(X, Y, minDistCond=10):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/07/13
    Modified: 2019/07/13
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Calculating the longest common susequences.
    
    @Parameters:
    ----------
    X: list-like or numpy array-like
        Time series X.
    
    Y: list-like or numpy array-like
        Time series Y.
    
    minDistCond, minPts: float
        dist <= minDistCond, then 2 points can
        be viewed as the same.
    
    @Return:
    ----------
    edit distance between series X and Y.
    '''
    length_X, length_Y = len(X), len(Y)
    
    # Initializing the dp matrix
    dp = np.zeros((length_X + 1, length_Y + 1))
    dp[0, :] = np.arange(0, length_Y + 1)
    dp[:, 0] = np.arange(0, length_X + 1)
    
    # Calculate the dp matrix
    for i in range(1, length_X + 1):
        for j in range(1, length_Y + 1):
            pointDist = np.abs(X[i-1] - Y[j-1])
            if pointDist <= minDistCond:
                subcost = 0
            else:
                subcost = 2
            dp[i, j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + subcost)
    return dp[-1, -1]


def fast_dynamic_time_warping(X, Y, radius=2):
    return fastdtw.fastdtw(X, Y, radius=radius)[0]


def get_adjacency_matrix_dtw(data, eta=0.5, verboseRound=500):
    if len(data) == 0:
        return None
    
    # Initializing the similarity matrix
    sampleNums = len(data)
    adjacencyMat = np.zeros((sampleNums, sampleNums))
    
    # Filling in the elements
    print("\n@Michael Yin: Start fastdtw calculating:")
    print("=====================================================")
    start = time.time()
    print("Start time : {}".format(datetime.now()))
    for i in range(sampleNums):
        if i % verboseRound == 0:
            print("    --Current fastdtw distance at {}, total is {}.".format(i, len(data)))
        for j in range(i, sampleNums):
            adjacencyMat[i, j] = dynamic_time_warping(data[i], data[j])
            adjacencyMat[j, i] = adjacencyMat[i, j]
    print("End time : {}".format(datetime.now()))
    end = time.time()
    print("Total time is {:5f}".format(end-start))
    print("=====================================================")

    # Gaussian transformation for the adjacency matrix
    adjacencyMat = np.exp( np.divide(-np.square(adjacencyMat), 2*eta**2) )
    return adjacencyMat


def get_adjacency_matrix_fastdtw(data, eta=0.5, radius=1, verboseRound=500):
    if len(data) == 0:
        return None
    
    # Initializing the similarity matrix
    sampleNums = len(data)
    adjacencyMat = np.zeros((sampleNums, sampleNums))
    
    # Filling in the elements
    print("\n@Michael Yin: Start fastdtw calculating:")
    print("=====================================================")
    start = time.time()
    print("Start time : {}".format(datetime.now()))
    for i in range(sampleNums):
        if i % verboseRound == 0:
            print("    --Current fastdtw distance at {}, total is {}.".format(i, len(data)))
        for j in range(i, sampleNums):
            adjacencyMat[i, j] = fast_dynamic_time_warping(data[i], data[j], radius=radius)
            adjacencyMat[j, i] = adjacencyMat[i, j]
    print("End time : {}".format(datetime.now()))
    end = time.time()
    print("Total time is {:5f}".format(end-start))
    print("=====================================================")

    # Gaussian transformation for the adjacency matrix
    adjacencyMat = np.exp( np.divide(-np.square(adjacencyMat), 2*eta**2) )
    return adjacencyMat


if __name__ == "__main__":
    pass

