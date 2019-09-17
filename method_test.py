#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:34:37 2019

@author: yinzhuo
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score

from UTILS import LoadSave
from UTILS import padding_seq

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


if __name__ == "__main__":
    currentData = load_data()
    groundTruth, ts, features = currentData[0], currentData[1], currentData[2]
    
    signal = {"current": ts}
    signal = pd.DataFrame(signal, columns=["current"])
    plotSave, plotShow = True, False
    ###############################################
    ###############################################
    
    ts_padding = padding_seq(ts, mode="pre_defined", padding_val=0, length=154)