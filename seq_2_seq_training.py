#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:16:14 2019

@author: yinzhuo
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import numpy as np
import scipy.sparse as scp
import seaborn as sns
import pandas as pd
from UTILS import LoadSave
#from keras.utils.vis_utils import plot_model

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, RepeatVector, Bidirectional, Masking, RNN, GRU, Lambda, TimeDistributed
from keras.models import Model, Sequential, model_from_json
from keras.layers.recurrent import GRUCell 
from keras.preprocessing.sequence import pad_sequences

from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)
colors = ["C" + str(i) for i in range(0, 9+1)]
markers = ["s", "^", "o", "d", "*"]
#mpl.rcParams["font.family"] = "Times New Roman"
###############################################################################
###############################################################################
#def batch_generator_seq(X=None, Y=None, batch_size=128,
#                        shuffle=True, seed=2019, steps=1):
#    '''
#    @Description:
#    ----------
#    Batch generator for the seq2seq task. The generator aim to generate the data
#    in teacher-forcing form.
#
#    @Parameters:
#    ----------
#    X: array-like
#        Sequence array. The shape of the sequence array is :
#            (## Number of training instances,
#             ## Number of time steps,
#             ## Number of features)
#
#    Y: array-like
#        Sequence array. The shape of the sequence array is :
#            (## Number of training instances,
#             ## Number of time steps,
#             ## Number of features)
#
#    batch_size: int-like
#        Number of instances of each batch.
#
#    shuffle: bool-like
#        Whether to shuffle the X data.
#
#    seed: int-like
#        Random seed.
#    
#    steps: int-like
#        How many step you want to predict.
#    
#    @Return:
#    ----------
#    Batch of data.
#
#    @BugTobeFixed:
#    ----------
#    None.
#    '''
#    number_of_batches = np.ceil(X.shape[0] / batch_size)
#    counter = 0
#    sample_index = np.arange(X.shape[0])
#    if shuffle:
#        np.random.seed(seed)
#        np.random.shuffle(sample_index)
#    
#    # Support the CSR format matrix
#    sparse = False
#    if scp.issparse(X):
#        sparse = True
#    
#    while True:
#        batch_index = sample_index[(batch_size * counter):(batch_size * (counter + 1))]
#        if sparse:
#            x_batch = X[batch_index, :, :].toarray()
#            y_batch = Y[batch_index, :, :].toarray()
#        else:
#            x_batch = X[batch_index, :, :]
#            y_batch = Y[batch_index, :, :]
#        counter += 1
#        
#        encoder_inputs = x_batch[::-1]
##        decoder_inputs = np.roll(y_batch, shift=1, axis=1)
##        decoder_inputs[:, 0, :] = 0
##        
##        decoder_outputs = np.roll(y_batch, shift=-(steps-1), axis=1)
##        decoder_outputs[:, -steps:, :] = 0
#        
#        # Prepare the decoder inputs and outpus
#        decoder_outputs = y_batch
#        decoder_inputs = np.zeros((decoder_outputs.shape[0], decoder_outputs.shape[1], 1))
#        
#        # Return the generator
#        yield ([encoder_inputs, decoder_inputs], decoder_outputs)
#
#        # Reset the count variable, which means the same batch will
#        # be generated in the next epoch.
#        if (counter == number_of_batches):
#            if shuffle:
#                np.random.shuffle(sample_index)
#            counter = 0
#
#
#def save_model(model=None, out_dir=None, f_arch='model_arch.png', f_model='model_arch.json', f_weights='model_weights.h5'):
#    '''
#    @Description:
#    ----------
#    Saves the Keras model description and model weights.
#
#    @Parameters:
#    ----------
#    model: keras model-like
#        A keras model.
#
#    out_dir: str-like
#        Directory to save model architecture and weights to.
#
#    f_arch: str-like
#        Filename for the model structure plot.(Disabled, it main cause consistency problem.)
#
#    f_model: str-like
#        Filename for model architecture.
#
#    f_weights: str-like
#        Filename for model weights.
#
#    @Return:
#    ----------
#    None.
#
#    @BugTobeFixed:
#    ----------
#    1. Calling Keras function plot_model will plot the model architecture, which requires the
#       "pydot" package and may cause the consistency problem.
#    '''
#    model.summary()
#    #plot_model(model, to_file=out_dir + f_arch)
#    json_string = model.to_json()
#    open(out_dir + f_model, 'w').write(json_string)
#    model.save_weights(out_dir + f_weights, overwrite=True)
#    
#    
#def load_model(dir_name=None, f_model='model_arch.json', f_weights='model_weights.h5' ):
#    '''
#    @Description:
#    ----------
#    Load a Keras model from disk to memory.
#
#    @Parameters:
#    ----------
#    dir_name: str-like
#        Directory in which the model architecture and weight files are present
#
#    f_model: keras model-like
#        File name for model architecture
#
#    f_weights: keras weight-like
#        Filename for model weights
#
#    @Return:
#    ----------
#    keras model.
#    '''
#
#    json_string = open(dir_name + f_model, 'r').read()
#    model = model_from_json(json_string)
#    
#    model.load_weights(f_weights)
#    return model

###############################################################################
#class seq2seq(object):
#    def __init__(self, layers=[], input_length=10, output_length=10,
#                 input_dim=1, output_dim=1, predict_steps=1, 
#                 learning_rate=0.01, decay=0, optimizer="adam", loss="mae",
#                 nb_epoch=2, batch_size=128, early_stop_rounds=5,
#                 regulariser_l1=0.000001, regulariser_l2=None, dropout=0.01,
#                 verbose=1):
#        
#        assert isinstance(layers, list), "Invalid layers parameters !"
#        self.layers = layers
#        self.input_length, self.output_length = input_length, output_length
#        self.input_dim, self.output_dim = input_dim, output_dim
#        self.predict_steps = predict_steps
#        
#        # Learning parameters
#        self.learning_rate, self.decay = learning_rate, decay
#        self.optimizer, self.loss_fcn = optimizer, loss
#        self.nb_epoch, self.batch_size = nb_epoch, batch_size
#        self.early_stop_rounds = early_stop_rounds
#        self.regulariser_l1, self.regulariser_l2 = regulariser_l1, regulariser_l2
#        self.dropout = dropout
#        self.verbose = verbose
#        
#        
#    def fit(self, data_in, data_val, data_test, dir_out=".//Models//",
#            write_model=True, model_layers=None):
#        
#        encoder_inputs, encoder, encoder_states = self._bulid_encoder(self.layers,
#                                                                      self.input_length,
#                                                                      self.input_dim)
#        decoder_inputs, decoder_outputs, decoder, decoder_dense = self._build_decoder(self.layers,
#                                                                                      self.input_length,
#                                                                                      self.input_dim,
#                                                                                      encoder_states)
#        
#        # Bulid the model and complie it
#        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
#        model.compile(optimizer=self.optimizer, loss=self.loss_fcn)
#        print("\n@Model summary:")
#        print("==============================================================")
#        model.summary()
#        print("==============================================================")
#        
#        # Fit the seq2seq model
#        early_stopping = EarlyStopping(monitor='val_loss',
#                                       patience=self.early_stop_rounds,
#                                       verbose=1)
#        model.fit_generator(generator=batch_generator_seq(data_in, data_in,
#                                                          batch_size=self.batch_size,
#                                                          shuffle=True, 
#                                                          steps=self.predict_steps),
#        nb_epoch=self.nb_epoch,
#        steps_per_epoch=np.ceil(data_in.shape[0] / self.batch_size),
#        verbose=self.verbose,
#        validation_data=batch_generator_seq(data_val, data_val,
#                                            batch_size=self.batch_size,
#                                            shuffle=True,
#                                            steps=self.predict_steps),
#        validation_steps=np.ceil(data_val.shape[0] / self.batch_size),
#        callbacks=[early_stopping])
#        
#        # Build the prediction model for prediction
#        predictors = self._build_predictor(layers=self.layers,
#                                           input_length=self.input_length,
#                                           input_dim=self.input_dim,
#                                           encoder=encoder,
#                                           encoder_inputs=encoder_inputs,
#                                           encoder_states=encoder_states,
#                                           decoder=decoder,
#                                           decoder_inputs=decoder_inputs,
#                                           decoder_dense=decoder_dense)
#        self.encoder_predict_model = predictors[0]
#        self.decoder_predict_model = predictors[1]
#        
#        
#    def predict(self, data_in, num_steps_to_predict=20, batch_size=128):
#        return self._predict(data_in, self.encoder_predict_model,
#                             self.decoder_predict_model, 
#                             num_steps_to_predict, batch_size)
#        
#        
#    def _bulid_encoder(self, layers=None, input_length=None, input_dim=None,
#                       kernel_regulariser=None, recurrent_regularizer=None, 
#                       bias_regulariser=None):
#        '''
#        @Description:
#        ----------
#        
#    
#        @Parameters:
#        ----------
#        
#        @Return:
#        ----------
#        
#        '''        
#
#        if input_length == None or input_dim == None:
#            raise ValueError("Invalid Encoder Input Parameters !")
#        
#        # Input of the encoder
#        encoder_inputs = Input(shape=(input_length, input_dim))
#        
#        # Construct the encoder structure
#        encoder_cells = []
#        for hidden_neurons in layers:
#            encoder_cells.append(GRUCell(hidden_neurons,
#                                         activation='elu',
#                                         kernel_regularizer=kernel_regulariser,
#                                         recurrent_regularizer=recurrent_regularizer,
#                                         bias_regularizer=bias_regulariser))
#        encoder = RNN(encoder_cells, return_state=True)
#        
#        # We only keep the hidden state of the last step of each cell
#        encoder_outputs_and_states = encoder(encoder_inputs)
#        encoder_states = encoder_outputs_and_states[1:]
#        return encoder_inputs, encoder, encoder_states
#    
#    
#    def _build_decoder(self, layers=None, output_length=None, output_dim=None,
#                       encoder_states=None,
#                       kernel_regulariser=None, recurrent_regularizer=None, 
#                       bias_regulariser=None, kernel_regularizer_dense=None, 
#                       bias_regularizer_dense=None):
#        '''
#        @Description:
#        ----------
#        
#        @Parameters:
#        ----------
#        
#        @Return:
#        ----------
#        
#        '''        
#
#        if output_length == None or output_dim == None:
#            raise ValueError("Invalid Encoder Input Parameters !")        
#        
#        # Construct the input of the decoder
#        decoder_inputs = Input(shape=(None, 1))
#        
#        # Construct the decoder structure 
#        decoder_cells = []
#        for hidden_neurons in layers:
#            decoder_cells.append(GRUCell(hidden_neurons,
#                                         activation='tanh',
#                                         kernel_regularizer=kernel_regulariser,
#                                         recurrent_regularizer=recurrent_regularizer,
#                                         bias_regularizer=bias_regulariser))
#        decoder = RNN(decoder_cells, return_sequences=True, return_state=True)
#        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
#        
#        # We only keep the hidden state tensor for every time step.
#        decoder_outputs = decoder_outputs_and_states[0]
#        
#        # Construct the dense for the regression
#        decoder_dense = Dense(output_dim,
#                              activation='sigmoid',
#                              kernel_regularizer=kernel_regularizer_dense,
#                              bias_regularizer=bias_regularizer_dense)
#        
#        decoder_outputs = decoder_dense(decoder_outputs)
#        return decoder_inputs, decoder_outputs, decoder, decoder_dense
#        
#    
#    def _build_predictor(self, layers=None, input_length=None, input_dim=None,
#                         encoder=None, encoder_inputs=None, encoder_states=None,
#                         decoder=None, decoder_inputs=None,
#                         decoder_dense=None):
#        '''
#        @Description:
#        ----------
#        
#    
#        @Parameters:
#        ----------
#        
#    
#        @Return:
#        ----------
#        
#        '''    
#        encoder_predict_model = Model(encoder_inputs, encoder_states)
#        
#        # Rebuild the prediction model
#        decoder_states_inputs = []
#        for hidden_neurons in layers:
#            decoder_states_inputs.append(Input(shape=(hidden_neurons, )))
#        
#        # Build the prediction structure
#        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)
#        
#        decoder_outputs = decoder_outputs_and_states[0]
#        decoder_outputs = decoder_dense(decoder_outputs)
#        
#        decoder_states = decoder_outputs_and_states[1:]
#        
#        decoder_predict_model = Model([decoder_inputs] + decoder_states_inputs,
#                                      [decoder_outputs] + decoder_states)
#        return encoder_predict_model, decoder_predict_model
#        
#    
#    def input_params_check(self, input_length=10, output_length=10, input_dim=1, output_dim=1,):
#        pass
#
#
#    def _predict(self, data_in, ecoder_predict_model=None,
#                 decoder_predict_model=None,
#                 num_steps_to_predict=None,
#                 batch_size=128):
#        '''
#        @Description:
#        ----------
#        Predict the output value for the data_in.
#    
#        @Parameters:
#        ----------
#        ecoder_predict_model: keras-model-like
#            Pretrained encoding keras model. Input is the sequence with the shape:
#            (## Number of training instances,
#             ## Number of time steps,
#             ## Number of features)
#            
#            The output of the model is the last hidden state of the ecoder.
#            
#        decoder_predict_model: keras model-like
#            Pretrained decoding keras model. The model has 2 input:
#            (1) The instance to be predicted.
#            (2) Hidden states from the encoder.
#            
#            So the shape of the input should be:
#            (## Number of training instances,
#             ## 1,
#             ## Number of features)
#             
#        num_steps_to_predict: int-like
#            How many steps we want to predict.
#    
#        @Return:
#        ----------
#        y_predicted: output time series for shape (batch_size, target_sequence_length,
#                                                   ouput_dimension)
#        '''       
#        y_predicted = []
#        data_in_reverse = data_in[:, ::-1, :]
#        
#        # Predict the hidded states of the decoder layer
#        states = ecoder_predict_model.predict(data_in_reverse)
#        
#        # The states must be a list
#        if not isinstance(states, list):
#            states = [states]
#
#        # Generate first value of the decoder input sequence
#        decoder_input = np.zeros((data_in.shape[0], 1, 1))
#        
#        # Generate first value of the decoder input sequence
#        for ind in range(num_steps_to_predict):
#            outputs_and_states = decoder_predict_model.predict([decoder_input] + states,
#                                                               batch_size=256, verbose=0)
##            decoder_input = outputs_and_states[0]
##            decoder_input = data_in_reverse[:, ind, :].reshape((len(data_in), 1, 1))
#            states = outputs_and_states[1:]
#            
#            # Record the outputs
#            y_predicted.append(outputs_and_states[0])
#            
#        return np.concatenate(y_predicted, axis=1)

#def lstm_autoencoder(data_train=None, data_valid=None, time_steps=None):
#    
#    # define model
#    model = Sequential()
#    model.add(Masking(mask_value=-5, input_shape=(data_train.shape[1], data_train.shape[2])))
#    model.add(LSTM(60, activation='tanh', input_shape=(data_train.shape[1], data_train.shape[2])))
#    model.add(RepeatVector(time_steps))
#    model.add(LSTM(100, activation='tanh', return_sequences=True))
#    model.add(TimeDistributed(Dense(data_train.shape[2], activation='sigmoid')))
#    model.add(TimeDistributed(Masking(mask_value=-5)))
#    model.compile(optimizer='adam', loss='mae')
#    
#    model.summary()
#    early_stopping = EarlyStopping(monitor='val_loss',
#                                   patience=50,
#                                   verbose=1)
#    # fit model
#    model.fit(data_train[:, ::-1, :], data_train,
#              validation_data=[data_valid[:, ::-1, :], data_valid],
#              epochs=200, verbose=1, batch_size=512, callbacks=[early_stopping])
#    
#    # demonstrate recreation
#    yhat = model.predict(data_valid[:, ::-1, :], verbose=0)
#    return yhat


def mask_loss_mae(y_true=[], y_pred=[], mask_val=-10):
    # Change the dtype to the default dtype
    mask_val = K.variable(mask_val, K.floatx())
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    
    # Calculating the mask loss
    mask = K.all(K.equal(y_true, mask_val), axis=-1, keepdims=True)
    mask = 1 - K.cast(mask, K.floatx())
    loss = K.abs(y_true - y_pred) * mask
    return K.sum(loss) / K.sum(mask)


def gru_bidirectional_autoencoder(data_train=None, data_valid=None,
                                  data_train_target=None, data_valid_target=None,
                                  time_steps=None, feature_num=None, 
                                  mask_val=-10, save_rep=True, path=".//Data//rep_data//"):
    # Data format
    data_train_reverse = data_train[:, ::-1, :]
    data_valid_reverse = data_valid[:, ::-1, :]
    
    # Setting some important parameters
    encoder_normal_neurons = 100
    hidden_states = 60

    '''
    Part 1: Double input of the encoder
    '''
    K.clear_session()
    encoder_inputs = Input(shape=(time_steps, feature_num))
    mask_layer = Masking(mask_value=mask_val)(encoder_inputs)
    encoder_gru = Bidirectional(GRU(encoder_normal_neurons,
                                    return_state=True, activation='elu'))
    encoder_outputs, forward_h, backward_h = encoder_gru(mask_layer)
    
    encoder_hidden_states = [forward_h, backward_h]
    encoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(encoder_hidden_states)
    states = Dense(hidden_states, activation='elu')(encoder_outputs)
    
    states_to_preserve = [states]

    '''
    Part 2: Decoder part. Set up the decoder,
            which will only process one timestep at a time.
    '''
    decoder_inputs = Input(shape=(1, 1))
    decoder = GRU(hidden_states, return_sequences=True, return_state=True)
    decoder_dense = Dense(1, activation='elu')
    
    decoder_all_outputs = []
    inputs = decoder_inputs
    for _ in range(time_steps):
        # Run the decoder on one timestep
        outputs, state_h = decoder(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        
        # Store the current prediction
        decoder_all_outputs.append(outputs)
        
        inputs = outputs
        states = [state_h]
    
    # Form the decoder outputs
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_all_outputs)
    
    '''
    Part 3: Define and complie the model.
    '''
    # Define and compile model as previously
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss=mask_loss_mae)
    
    '''
    Part 4: Fit the model.
    '''
    decoder_input_train = np.zeros((len(data_train), 1, 1))
    decoder_input_train[:, 0, 0] = data_train_reverse[:, 0, 0]
    decoder_input_valid = np.zeros((len(data_valid), 1, 1))
    decoder_input_valid[:, 0, 0] = data_valid_reverse[:, 0, 0]
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=500,
                                   verbose=1)
    model.fit([data_train_reverse, decoder_input_train], data_train_target,
              batch_size=len(data_train),
              epochs=4000, verbose=1,
              validation_data=[[data_valid_reverse,
                                decoder_input_valid], data_valid_target],
              callbacks=[early_stopping])
    
    ###############################################
    ###############################################        
    '''
    Part 5: Construct the encoder. Generating the predictions for evaluation.
    '''
    decoder_input_data = np.zeros((len(data_valid), 1, 1))
    decoder_input_data[:, 0, 0] = 0
    data_valid_recon = model.predict([data_valid_reverse,
                                      decoder_input_data])
    
    encoder = Model(encoder_inputs, states_to_preserve[0])
    
    dense_train = encoder.predict(data_train_reverse)
    dense_valid = encoder.predict(data_valid_reverse)
    repTs = np.concatenate([dense_train, dense_valid], axis=0)
    
    if save_rep:
        ls = LoadSave(path + "rnn_rep_gru_bidirectional_" + str(hidden_states) + ".pkl")
        ls.save_data(repTs)

    return data_valid_recon, repTs


def lstm_autoencoder_repeatvector(data_train=None, data_valid=None,
                                  data_train_target=None, data_valid_target=None,
                                  time_steps=None, feature_num=None,
                                  mask_val=-10, save_rep=True, path=".//Data//rep_data//"):
    # Parameters
    encoder_hidden = 80
    decoder_hidden = 100
    data_train_reverse = data_train[:, ::-1, :]
    data_valid_reverse = data_valid[:, ::-1, :]
        
    
    # define model
    model = Sequential()
    model.add(Masking(mask_value=mask_val,
                      input_shape=(time_steps, feature_num)))
    model.add(LSTM(encoder_hidden, activation='elu',
                   input_shape=(time_steps, feature_num)))
    
    model.add(RepeatVector(time_steps))
    model.add(LSTM(decoder_hidden, activation='elu',
                   return_sequences=True))
    model.add(TimeDistributed(Dense(feature_num,
                                    activation='linear')))
    
    model.compile(optimizer='adam', loss=mask_loss_mae)
    
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=500,
                                   verbose=1)
    # fit model
    model.fit(data_train_reverse, data_train_target,
              validation_data=[data_valid_reverse, data_valid_target],
              epochs=5000, verbose=1, batch_size=data_train.shape[0],
              callbacks=[early_stopping])

    ###############################################
    ###############################################
    '''
    Part 5: Construct the encoder. Generating the predictions for evaluation.
    '''
    encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)
    
    data_valid_recon = model.predict(data_valid_reverse)
    
    dense_train = encoder.predict(data_train_reverse)
    dense_valid = encoder.predict(data_valid_reverse)
    repTs = np.concatenate([dense_train, dense_valid], axis=0)
    
    if save_rep:
        ls = LoadSave(path + "rnn_rep_lstm_repvec_" + str(encoder_hidden) + ".pkl")
        ls.save_data(repTs)
    
    return data_valid_recon, repTs


def gru_double_input_autoencoder(data_train=None, data_valid=None,
                                 data_train_target=None, data_valid_target=None,
                                 time_steps=None, feature_num=None,
                                 mask_val=-10, save_rep=True, path=".//Data//rep_data//"):
    
    # Data format
    data_train_reverse = data_train[:, ::-1, :]
    data_valid_reverse = data_valid[:, ::-1, :]
    
    # Setting some important parameters
    encoder_nromal_neurons = 100
    encoder_reverse_neurons = 100
    
    hidden_states = 90
    
    '''
    Part 1: Double input of the encoder
    '''
    K.clear_session()
    encoder_normal_inputs = Input(shape=(time_steps, feature_num))
    mask_normal = Masking(mask_value=mask_val)(encoder_normal_inputs)
    encoder_normal = GRU(encoder_nromal_neurons, return_state=True, activation='tanh')
    encoder_normal_outputs, state_normal_h = encoder_normal(mask_normal)

    encoder_reverse_inputs = Input(shape=(time_steps, feature_num))
    mask_reverse = Masking(mask_value=mask_val)(encoder_reverse_inputs)
    encoder_reverse = GRU(encoder_reverse_neurons, return_state=True, activation='tanh')
    encoder_reverse_outputs, state_reverse_h = encoder_reverse(mask_reverse)
    
    encoder_hidden_states = [state_normal_h, state_reverse_h]
    encoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(encoder_hidden_states)
    states = Dense(hidden_states, activation='elu')(encoder_outputs)
    
    states_to_preserve = [states]
    
    '''
    Part 2: Decoder part. Set up the decoder,
            which will only process one timestep at a time.
    '''
    decoder_inputs = Input(shape=(1, 1))
    decoder = GRU(hidden_states, return_sequences=True, return_state=True)
    decoder_dense = Dense(1, activation='sigmoid')
    
    decoder_all_outputs = []
    inputs = decoder_inputs
    for _ in range(time_steps):
        # Run the decoder on one timestep
        outputs, state_h = decoder(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        
        # Store the current prediction
        decoder_all_outputs.append(outputs)
        
        inputs = outputs
        states = [state_h]
    
    # Form the decoder outputs
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_all_outputs)
    
    '''
    Part 3: Define and complie the model.
    '''
    # Define and compile model as previously
    model = Model([encoder_normal_inputs, encoder_reverse_inputs,
                   decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss=mask_loss_mae)
    
    '''
    Part 4: Fit the model.
    '''
    decoder_input_train = np.zeros((len(data_train), 1, 1))
    decoder_input_train[:, 0, 0] = data_train_reverse[:, 0, 0]
    decoder_input_valid = np.zeros((len(data_valid), 1, 1))
    decoder_input_valid[:, 0, 0] = data_valid_reverse[:, 0, 0]
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=500,
                                   verbose=1)
    model.fit([data_train, data_train_reverse, decoder_input_train], data_train_target,
              batch_size=len(data_train),
              epochs=4000, verbose=1,
              validation_data=[[data_valid,
                                data_valid_reverse,
                                decoder_input_valid], data_valid_target],
              callbacks=[early_stopping])
    
    ###############################################
    ###############################################        
    '''
    Part 5: Construct the encoder. Generating the predictions for evaluation.
    '''
    decoder_input_data = np.zeros((len(data_valid), 1, 1))
    decoder_input_data[:, 0, 0] = 0
    data_valid_recon = model.predict([data_valid, data_valid_reverse,
                                      decoder_input_data])
    
    encoder = Model([encoder_normal_inputs, encoder_reverse_inputs], states_to_preserve[0])
    
    dense_train = encoder.predict([data_train, data_train_reverse])
    dense_valid = encoder.predict([data_valid, data_valid_reverse])
    repTs = np.concatenate([dense_train, dense_valid], axis=0)
    
    if save_rep:
        ls = LoadSave(path + "rnn_rep_gru_twoinput_" + str(hidden_states) + ".pkl")
        ls.save_data(repTs)

    return data_valid_recon, repTs

###############################################################################
###############################################################################
def load_data():
    fileName = ["current_ts_merged_3300.pkl"]
    ls = LoadSave()
    currentData = ls.load_data(".//Data//" + fileName[0])
    return currentData


def select_best_lof_value(data=None, y_true=None,
                          nn_range=[i for i in range(60, 300, 10)]):
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


# Basic lagging step
def lagging_features(data, name=None, laggingStep=[1, 2, 3]):
    assert name, "Invalid feature name!"
    for step in laggingStep:
        tmp = data[[name, "timeStep"]].copy()
        tmp.rename({name: name + "_lag_" + str(step)}, axis=1, inplace=True)
        tmp["timeStep"] += step
        data = pd.merge(data, tmp, on="timeStep", how="left")
    return data


# Statistical features
def statistical_features(data, name=None, operation=["mean", "std", "max"], timeRange=5):
    assert name, "Invalid feature name!"
    index = list(data.index)
    featureValues = data[name].values
    
    aggMax = []
    aggMean = []
    aggStd = []
    aggFirstLast = []
    for currInd in index:
        tmp = featureValues[max(0, currInd-timeRange):currInd]
        
        if len(tmp) == 0:
            aggMax.append(np.nan)
        else:
            aggMax.append(np.max(tmp))
        aggMean.append(np.nanmean(tmp))
        aggStd.append(np.nanstd(tmp))
        aggFirstLast.append(featureValues[currInd] - featureValues[max(0, currInd-timeRange)])
    
    data[name + "_lag_max_" + str(timeRange)] = aggMax
    data[name + "_lag_mean_" + str(timeRange)] = aggMean
    data[name + "_lag_std_" + str(timeRange)] = aggStd
    data[name + "_last_first_" + str(timeRange)] = aggFirstLast
    return data


def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = np.sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


def feature_engineering(ts_original):
    
    ts = pd.DataFrame(ts_original, columns=["val"])
    ts.reset_index(inplace=True)
    ts.rename({"index":"timeStep"}, axis=1, inplace=True)
    
    ts = lagging_features(ts, name="val", laggingStep=[1, 2, 3, 10])
#    ts = statistical_features(ts, name="val", timeRange=5)
    ts = statistical_features(ts, name="val", timeRange=10)
    
    return ts

###############################################################################
###############################################################################        
if __name__ == "__main__":
    plotShow, plotSave = False, True
    ###############################################
    ###############################################    
    '''
    Step 1: Loading the training data, and padding them.
    '''
    currentData = load_data()
    groundTruth, ts_original = currentData[0], currentData[1]
    y_true=np.where(groundTruth["label"] != -1, 1, -1)
    
    ts = [feature_engineering(item).drop(["timeStep"], axis=1).fillna(0) for ind, item in enumerate(ts_original)]
    index = [ind for ind, item in enumerate(ts_original)]
    ts_target = [item["val"].values for item in ts]
    ts = [item.values for item in ts]
    
    # Scaling the data
    #---------------------------------------------------------
    for ind in range(len(ts)):
        X_sc = MinMaxScaler()
        ts[ind] = X_sc.fit_transform(ts[ind])
        ts_target[ind] = X_sc.fit_transform(ts_target[ind].reshape((-1, 1)))
        
    paddingTs = pad_sequences(ts, padding="pre", dtype="float64", value=-10, maxlen=159)
    paddingTs_target = pad_sequences(ts_target, padding="pre", dtype="float64", value=-10, maxlen=159)
    
    # Spliting the training and testing data
    #---------------------------------------------------------    
    trainPrecent = 0.85
    train_index, valid_index = index[:int(trainPrecent * len(paddingTs))], index[int(trainPrecent * len(paddingTs)):]
    data_train, data_valid = paddingTs[:int(trainPrecent * len(paddingTs)), :], paddingTs[int(trainPrecent * len(paddingTs)):, :]
    data_train_target, data_valid_target = paddingTs_target[:int(trainPrecent * len(paddingTs)), :], paddingTs_target[int(trainPrecent * len(paddingTs)):, :]
    
    data_train_nums, data_valid_nums, ts_step_nums, ts_feature_nums = len(data_train), len(data_valid), data_train.shape[1], data_train.shape[2]
    ts_nums = len(ts)
    
    ###############################################
    ###############################################    
    '''
    Step 2: Training the GRU model.
    '''
#    predicted_res, rep = gru_bidirectional_autoencoder(data_train=data_train,
#                                                       data_valid=data_valid,
#                                                       data_train_target=data_train_target,
#                                                       data_valid_target=data_valid_target,
#                                                       time_steps=ts_step_nums,
#                                                       feature_num=ts_feature_nums,
#                                                       mask_val=-10)
    
    predicted_res, rep = gru_double_input_autoencoder(data_train=data_train,
                                                      data_valid=data_valid,
                                                      data_train_target=data_train_target,
                                                      data_valid_target=data_valid_target,
                                                      time_steps=ts_step_nums,
                                                      feature_num=ts_feature_nums,
                                                      mask_val=-10)
    
#    predicted_res, rep = lstm_autoencoder_repeatvector(data_train=data_train,
#                                                       data_valid=data_valid,
#                                                       data_train_target=data_train_target,
#                                                       data_valid_target=data_valid_target,
#                                                       time_steps=ts_step_nums,
#                                                       feature_num=ts_feature_nums,
#                                                       mask_val=-10)
    ###############################################
    ###############################################
    '''
    Step 3: Basic plots.
    '''
    
    # Plot 1: Prediction Plot.
    data_train_no_padding = [item[item[:, 0]!=-10, :] for item in data_train]
    data_valid_no_padding = [item[item[:, 0]!=-10, :] for item in data_valid]
    predicted_no_padding = [item[item[:, 0]>=0.0001, :] for item in predicted_res]
    
    plotInd = np.random.randint(0, len(data_valid), 20)
    plt.figure(figsize=(10, 4))
    for ind in plotInd:
        plt.plot(data_valid_no_padding[ind][:, 0], color='b', linewidth=2)
        plt.plot(predicted_no_padding[ind][:, 0], color='r', linewidth=2)
    plt.title("Validation Prediction")
    
    # Plot 2: Visualizing the 2-d distribution in feature space
    # -----------------------------------------------------------
    # kpca ===>> 2 components
    components = 2
    pca = KernelPCA(n_components=components, kernel='rbf')
    rep_mean = pca.fit_transform(rep)
    
    # Kernel PCA plots.
    plt.close("all")
    f, axObj = plt.subplots(figsize=(7, 6))
    uniqueLabels = [1, 2, 3, -1]
    for ind, label in enumerate(uniqueLabels):
        sampleIndex = np.arange(0, len(groundTruth))
        labeledSampleIndex = sampleIndex[groundTruth["label"] == label]
        
        coords_mean = rep_mean[labeledSampleIndex, :]
        if label != -1:
            axObj.scatter(coords_mean[:, 0], coords_mean[:, 1], s=8, alpha=0.4,
                 color=colors[ind], marker=markers[ind], label="Class " + str(label))
        else:
            axObj.scatter(coords_mean[:, 0], coords_mean[:, 1], s=8,
                 alpha=0.4, color='r', marker="x", label="Class Abnormal")
        axObj.tick_params(axis="both", labelsize=8)
        
        axObj.legend(fontsize=8)
        axObj.legend(fontsize=8)
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//5_RNNREP_kpca_rnn_rep_distribution.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
        
    # -----------------------------------------------------------
    # LOF for the anomaly detection
    rocBestInd, rocBest, rocRec = [], [], []
    for data_rep in [rep]:
        tmp_1, tmp_2, tmp_3 = select_best_lof_value(data_rep,
                                                    y_true=y_true,
                                                    nn_range=[i for i in range(60, 250, 10)])
        rocBestInd.append(tmp_1)
        rocBest.append(tmp_2)
        rocRec.append(tmp_3)
    
    fig, axObj = plt.subplots(2, 2, figsize=(15, 9))
    pts_index = [i for i in range(60, 250, 10)]
    lw = 2
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][0].plot(rocRec_ind[2][rocBestInd_ind][1],
                         rocRec_ind[2][rocBestInd_ind][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocBest_ind),
                         lw=lw)
    axObj[0][0].legend(fontsize=7)
    axObj[0][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    #---------------------
    index_0_1 = 0
    pts = pts_index[index_0_1]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[0][1].plot(rocRec_ind[2][index_0_1][1],
                         rocRec_ind[2][index_0_1][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_0_1]),
                         lw=lw)
    axObj[0][1].legend(fontsize=7)
    axObj[0][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    #---------------------
    index_1_0 = 10
    pts = pts_index[index_1_0]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][0].plot(rocRec_ind[2][index_1_0][1],
                         rocRec_ind[2][index_1_0][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_1_0]),
                         lw=lw)
    axObj[1][0].legend(fontsize=7)
    axObj[1][0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    #---------------------
    index_1_1 = 18
    pts = pts_index[index_1_1]
    for ind, (rocBestInd_ind, rocBest_ind, rocRec_ind) in enumerate(zip(rocBestInd, rocBest, rocRec)):
        axObj[1][1].plot(rocRec_ind[2][index_1_1][1],
                         rocRec_ind[2][index_1_1][2],
                         label='Rep_' + str(ind) + '(auc={:.5f})'.format(rocRec_ind[1][index_1_1]),
                         lw=lw)
    axObj[1][1].legend(fontsize=7)
    axObj[1][1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axObj[0][0].tick_params(axis="both", labelsize=8)
    
    for obj in axObj.ravel():
        obj.grid(False)
        obj.set_xlim(0, 1)
        obj.set_ylim(0, )
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//5_RNNREP_roc_auc_plot.png", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")

    # -----------------------------------------------------------
    # Heatmap of the rep
    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = pd.DataFrame(rep)
    featureCorr = tmp.corr()
#    ax = sns.heatmap(featureCorr, xticklabels=2, yticklabels=2,
#                 cmap="Blues", fmt='.2f', annot=True,
#                 annot_kws={'size':4.5,'weight':'bold'}, ax=ax)
    ax = sns.heatmap(featureCorr, cmap="Blues", ax=ax)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.tick_params(axis="x", labelsize=7)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    
    if plotSave:
        plt.savefig(".//Plots//5_RNNREP_feature_corr_plot", dpi=500, bbox_inches="tight")
        plt.close("all")
    if not plotShow:
        plt.close("all")
