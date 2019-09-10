#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:33:05 2019

@author: yinzhuo
"""
# Disable or enable the GPU computation
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pandas as pd
import numpy as np
import seaborn as sns
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from UTILS import ReduceMemoryUsage, LoadSave

from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Masking
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

np.random.seed(2019)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_data():
    fileName = ["current_ts_merged_3000.pkl"]
    ls = LoadSave()
    currentData = ls.load_data("..//Data//" + fileName[0])
    return currentData

def lstm_autoencoder(n_hids=10, ts=[]):
    pass



if __name__ == "__main__":
    currentData = load_data()
    groundTruth, ts = currentData[0], currentData[1]
    X_sc = MinMaxScaler()
    for ind in range(len(ts)):
        if max(ts[ind]) != min(ts[ind]):
            ts[ind] = (np.array(ts[ind]) - min(ts[ind]))/(max(ts[ind]) - min(ts[ind]))
    
    paddingTs = pad_sequences(ts, padding="post", dtype="float64", value=0)
    tsTotalNums, tsTotalPtsNums = len(ts), len(paddingTs[0])
    
    trainPrecent = 0.8
    data_train, data_valid = paddingTs[:int(trainPrecent * len(paddingTs)), :], paddingTs[int(trainPrecent * len(paddingTs)):, :]
    data_train_nums, data_valid_nums, ts_point_nums = len(data_train), len(data_valid), len(data_train[0])
    
    data_train = data_train.reshape((data_train_nums, ts_point_nums, 1))
    data_valid = data_valid.reshape((data_valid_nums, ts_point_nums, 1))
    
    data_train_to_predict = data_train[:, 1:, :]
    data_valid_to_predict = data_valid[:, 1:, :]
    
    n_embedding = 20
    # Input layer
    input_layer = Input(shape=(tsTotalPtsNums, 1), name="input_layer")
    
    # Mask layer for masking -65535
    mask_layer = Masking(mask_value=0, input_shape=(tsTotalPtsNums, 1), name="mask_layer")(input_layer)
    
    # Encoding layer
    encoder_layer = LSTM(n_embedding, activation="relu", name="encoder_layer")(mask_layer)
    
    # Define the reconstruct layer
    decoder_layer_reconstruct = RepeatVector(tsTotalPtsNums, name="rep_reconstruct")(encoder_layer)
    decoder_layer_reconstruct = LSTM(n_embedding, activation="relu", return_sequences=True, name="reconstruct")(decoder_layer_reconstruct)
    decoder_layer_reconstruct = TimeDistributed(Dense(1))(decoder_layer_reconstruct)
    
    # Define the predictive layer
    decoder_layer_prediction = RepeatVector(tsTotalPtsNums-1, name="rep_prediction")(encoder_layer)
    decoder_layer_prediction = LSTM(n_embedding, activation="relu", return_sequences=True, name="prediction")(decoder_layer_prediction)
    decoder_layer_prediction = TimeDistributed(Dense(1))(decoder_layer_prediction)
    
    model = Model(inputs=input_layer, outputs=[decoder_layer_reconstruct,
                                               decoder_layer_prediction])
    model.compile(optimizer="adam", loss="mse")
    print(model.summary())
    
    # Define the predictive layer
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    model.fit(x=data_train, y=[data_train, data_train_to_predict],
              validation_data=[data_valid, [data_valid, data_valid_to_predict]],
              epochs=100, verbose=1, batch_size=128, callbacks=[early_stopping])
    