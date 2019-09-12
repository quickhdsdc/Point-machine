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
from UTILS import LoadSave
#from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Masking, RNN
from keras.models import Model, Sequential, model_from_json
from keras.layers.recurrent import GRUCell 
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2019)
###############################################################################
###############################################################################
def batch_generator_seq(X=None, Y=None, batch_size=128,
                        shuffle=True, seed=2019, steps=1):
    '''
    @Description:
    ----------
    Batch generator for the seq2seq task. The generator aim to generate the data
    in teacher-forcing form.

    @Parameters:
    ----------
    X: array-like
        Sequence array. The shape of the sequence array is :
            (## Number of training instances,
             ## Number of time steps,
             ## Number of features)

    Y: array-like
        Sequence array. The shape of the sequence array is :
            (## Number of training instances,
             ## Number of time steps,
             ## Number of features)

    batch_size: int-like
        Number of instances of each batch.

    shuffle: bool-like
        Whether to shuffle the X data.

    seed: int-like
        Random seed.
    
    steps: int-like
        How many step you want to predict.
    
    @Return:
    ----------
    Batch of data.

    @BugTobeFixed:
    ----------
    None.
    '''
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sample_index)
    
    # Support the CSR format matrix
    sparse = False
    if scp.issparse(X):
        sparse = True
    
    while True:
        batch_index = sample_index[(batch_size * counter):(batch_size * (counter + 1))]
        if sparse:
            x_batch = X[batch_index, :, :].toarray()
            y_batch = Y[batch_index, :, :].toarray()
        else:
            x_batch = X[batch_index, :, :]
            y_batch = Y[batch_index, :, :]
        counter += 1
        
        encoder_inputs = x_batch[::-1]
#        decoder_inputs = np.roll(y_batch, shift=1, axis=1)
#        decoder_inputs[:, 0, :] = 0
#        
#        decoder_outputs = np.roll(y_batch, shift=-(steps-1), axis=1)
#        decoder_outputs[:, -steps:, :] = 0
        
        # Prepare the decoder inputs and outpus
        decoder_outputs = y_batch
        decoder_inputs = np.zeros((decoder_outputs.shape[0], decoder_outputs.shape[1], 1))
        
        # Return the generator
        yield ([encoder_inputs, decoder_inputs], decoder_outputs)

        # Reset the count variable, which means the same batch will
        # be generated in the next epoch.
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def save_model(model=None, out_dir=None, f_arch='model_arch.png', f_model='model_arch.json', f_weights='model_weights.h5'):
    '''
    @Description:
    ----------
    Saves the Keras model description and model weights.

    @Parameters:
    ----------
    model: keras model-like
        A keras model.

    out_dir: str-like
        Directory to save model architecture and weights to.

    f_arch: str-like
        Filename for the model structure plot.(Disabled, it main cause consistency problem.)

    f_model: str-like
        Filename for model architecture.

    f_weights: str-like
        Filename for model weights.

    @Return:
    ----------
    None.

    @BugTobeFixed:
    ----------
    1. Calling Keras function plot_model will plot the model architecture, which requires the
       "pydot" package and may cause the consistency problem.
    '''
    model.summary()
    #plot_model(model, to_file=out_dir + f_arch)
    json_string = model.to_json()
    open(out_dir + f_model, 'w').write(json_string)
    model.save_weights(out_dir + f_weights, overwrite=True)
    
    
def load_model(dir_name=None, f_model='model_arch.json', f_weights='model_weights.h5' ):
    '''
    @Description:
    ----------
    Load a Keras model from disk to memory.

    @Parameters:
    ----------
    dir_name: str-like
        Directory in which the model architecture and weight files are present

    f_model: keras model-like
        File name for model architecture

    f_weights: keras weight-like
        Filename for model weights

    @Return:
    ----------
    keras model.
    '''

    json_string = open(dir_name + f_model, 'r').read()
    model = model_from_json(json_string)
    
    model.load_weights(f_weights)
    return model

###############################################################################
class seq2seq(object):
    def __init__(self, layers=[], input_length=10, output_length=10,
                 input_dim=1, output_dim=1, predict_steps=1, 
                 learning_rate=0.01, decay=0, optimizer="adam", loss="mae",
                 nb_epoch=2, batch_size=128, early_stop_rounds=5,
                 regulariser_l1=0.000001, regulariser_l2=None, dropout=0.01,
                 verbose=1):
        
        assert isinstance(layers, list), "Invalid layers parameters !"
        self.layers = layers
        self.input_length, self.output_length = input_length, output_length
        self.input_dim, self.output_dim = input_dim, output_dim
        self.predict_steps = predict_steps
        
        # Learning parameters
        self.learning_rate, self.decay = learning_rate, decay
        self.optimizer, self.loss_fcn = optimizer, loss
        self.nb_epoch, self.batch_size = nb_epoch, batch_size
        self.early_stop_rounds = early_stop_rounds
        self.regulariser_l1, self.regulariser_l2 = regulariser_l1, regulariser_l2
        self.dropout = dropout
        self.verbose = verbose
        
        
    def fit(self, data_in, data_val, data_test, dir_out=".//Models//",
            write_model=True, model_layers=None):
        
        encoder_inputs, encoder, encoder_states = self._bulid_encoder(self.layers,
                                                                      self.input_length,
                                                                      self.input_dim)
        decoder_inputs, decoder_outputs, decoder, decoder_dense = self._build_decoder(self.layers,
                                                                                      self.input_length,
                                                                                      self.input_dim,
                                                                                      encoder_states)
        
        # Bulid the model and complie it
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        model.compile(optimizer=self.optimizer, loss=self.loss_fcn)
        print("\n@Model summary:")
        print("==============================================================")
        model.summary()
        print("==============================================================")
        
        # Fit the seq2seq model
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.early_stop_rounds,
                                       verbose=1)
        model.fit_generator(generator=batch_generator_seq(data_in, data_in,
                                                          batch_size=self.batch_size,
                                                          shuffle=True, 
                                                          steps=self.predict_steps),
        nb_epoch=self.nb_epoch,
        steps_per_epoch=np.ceil(data_in.shape[0] / self.batch_size),
        verbose=self.verbose,
        validation_data=batch_generator_seq(data_val, data_val,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            steps=self.predict_steps),
        validation_steps=np.ceil(data_val.shape[0] / self.batch_size),
        callbacks=[early_stopping])
        
        # Build the prediction model for prediction
        predictors = self._build_predictor(layers=self.layers,
                                           input_length=self.input_length,
                                           input_dim=self.input_dim,
                                           encoder=encoder,
                                           encoder_inputs=encoder_inputs,
                                           encoder_states=encoder_states,
                                           decoder=decoder,
                                           decoder_inputs=decoder_inputs,
                                           decoder_dense=decoder_dense)
        self.encoder_predict_model = predictors[0]
        self.decoder_predict_model = predictors[1]
        
        
    def predict(self, data_in, num_steps_to_predict=20, batch_size=128):
        return self._predict(data_in, self.encoder_predict_model,
                             self.decoder_predict_model, 
                             num_steps_to_predict, batch_size)
        
        
    def _bulid_encoder(self, layers=None, input_length=None, input_dim=None,
                       kernel_regulariser=None, recurrent_regularizer=None, 
                       bias_regulariser=None):
        '''
        @Description:
        ----------
        
    
        @Parameters:
        ----------
        
        @Return:
        ----------
        
        '''        

        if input_length == None or input_dim == None:
            raise ValueError("Invalid Encoder Input Parameters !")
        
        # Input of the encoder
        encoder_inputs = Input(shape=(input_length, input_dim))
        
        # Construct the encoder structure
        encoder_cells = []
        for hidden_neurons in layers:
            encoder_cells.append(GRUCell(hidden_neurons,
                                         activation='elu',
                                         kernel_regularizer=kernel_regulariser,
                                         recurrent_regularizer=recurrent_regularizer,
                                         bias_regularizer=bias_regulariser))
        encoder = RNN(encoder_cells, return_state=True)
        
        # We only keep the hidden state of the last step of each cell
        encoder_outputs_and_states = encoder(encoder_inputs)
        encoder_states = encoder_outputs_and_states[1:]
        return encoder_inputs, encoder, encoder_states
    
    
    def _build_decoder(self, layers=None, output_length=None, output_dim=None,
                       encoder_states=None,
                       kernel_regulariser=None, recurrent_regularizer=None, 
                       bias_regulariser=None, kernel_regularizer_dense=None, 
                       bias_regularizer_dense=None):
        '''
        @Description:
        ----------
        
        @Parameters:
        ----------
        
        @Return:
        ----------
        
        '''        

        if output_length == None or output_dim == None:
            raise ValueError("Invalid Encoder Input Parameters !")        
        
        # Construct the input of the decoder
        decoder_inputs = Input(shape=(None, 1))
        
        # Construct the decoder structure 
        decoder_cells = []
        for hidden_neurons in layers:
            decoder_cells.append(GRUCell(hidden_neurons,
                                         activation='tanh',
                                         kernel_regularizer=kernel_regulariser,
                                         recurrent_regularizer=recurrent_regularizer,
                                         bias_regularizer=bias_regulariser))
        decoder = RNN(decoder_cells, return_sequences=True, return_state=True)
        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
        
        # We only keep the hidden state tensor for every time step.
        decoder_outputs = decoder_outputs_and_states[0]
        
        # Construct the dense for the regression
        decoder_dense = Dense(output_dim,
                              activation='sigmoid',
                              kernel_regularizer=kernel_regularizer_dense,
                              bias_regularizer=bias_regularizer_dense)
        
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_outputs, decoder, decoder_dense
        
    
    def _build_predictor(self, layers=None, input_length=None, input_dim=None,
                         encoder=None, encoder_inputs=None, encoder_states=None,
                         decoder=None, decoder_inputs=None,
                         decoder_dense=None):
        '''
        @Description:
        ----------
        
    
        @Parameters:
        ----------
        
    
        @Return:
        ----------
        
        '''    
        encoder_predict_model = Model(encoder_inputs, encoder_states)
        
        # Rebuild the prediction model
        decoder_states_inputs = []
        for hidden_neurons in layers:
            decoder_states_inputs.append(Input(shape=(hidden_neurons, )))
        
        # Build the prediction structure
        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)
        
        decoder_outputs = decoder_outputs_and_states[0]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        decoder_states = decoder_outputs_and_states[1:]
        
        decoder_predict_model = Model([decoder_inputs] + decoder_states_inputs,
                                      [decoder_outputs] + decoder_states)
        return encoder_predict_model, decoder_predict_model
        
    
    def input_params_check(self, input_length=10, output_length=10, input_dim=1, output_dim=1,):
        pass


    def _predict(self, data_in, ecoder_predict_model=None,
                 decoder_predict_model=None,
                 num_steps_to_predict=None,
                 batch_size=128):
        '''
        @Description:
        ----------
        Predict the output value for the data_in.
    
        @Parameters:
        ----------
        ecoder_predict_model: keras-model-like
            Pretrained encoding keras model. Input is the sequence with the shape:
            (## Number of training instances,
             ## Number of time steps,
             ## Number of features)
            
            The output of the model is the last hidden state of the ecoder.
            
        decoder_predict_model: keras model-like
            Pretrained decoding keras model. The model has 2 input:
            (1) The instance to be predicted.
            (2) Hidden states from the encoder.
            
            So the shape of the input should be:
            (## Number of training instances,
             ## 1,
             ## Number of features)
             
        num_steps_to_predict: int-like
            How many steps we want to predict.
    
        @Return:
        ----------
        y_predicted: output time series for shape (batch_size, target_sequence_length,
                                                   ouput_dimension)
        '''       
        y_predicted = []
        data_in_reverse = data_in[:, ::-1, :]
        
        # Predict the hidded states of the decoder layer
        states = ecoder_predict_model.predict(data_in_reverse)
        
        # The states must be a list
        if not isinstance(states, list):
            states = [states]

        # Generate first value of the decoder input sequence
        decoder_input = np.zeros((data_in.shape[0], 1, 1))
        
        # Generate first value of the decoder input sequence
        for ind in range(num_steps_to_predict):
            outputs_and_states = decoder_predict_model.predict([decoder_input] + states,
                                                               batch_size=256, verbose=0)
#            decoder_input = outputs_and_states[0]
#            decoder_input = data_in_reverse[:, ind, :].reshape((len(data_in), 1, 1))
            states = outputs_and_states[1:]
            
            # Record the outputs
            y_predicted.append(outputs_and_states[0])
            
        return np.concatenate(y_predicted, axis=1)

def lstm_autoencoder(data_train=None, data_valid=None, time_steps=None):
    
    # define model
    model = Sequential()
    model.add(Masking(mask_value=-5, input_shape=(data_train.shape[1], data_train.shape[2])))
    model.add(LSTM(60, activation='tanh', input_shape=(data_train.shape[1], data_train.shape[2])))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(data_train.shape[2], activation='sigmoid')))
    model.add(TimeDistributed(Masking(mask_value=-5)))
    model.compile(optimizer='adam', loss='mae')
    model.build(data_train.shape)
    
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=50,
                                   verbose=1)
    # fit model
    model.fit(data_train[:, ::-1, :], data_train,
              validation_data=[data_valid[:, ::-1, :], data_valid],
              epochs=200, verbose=1, batch_size=512, callbacks=[early_stopping])
    
    # demonstrate recreation
    yhat = model.predict(data_valid[:, ::-1, :], verbose=0)
    return yhat

###############################################################################
def load_data():
    fileName = ["current_ts_merged_3300.pkl"]
    ls = LoadSave()
    currentData = ls.load_data(".//Data//" + fileName[0])
    return currentData

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


if __name__ == "__main__":
    currentData = load_data()
    
    '''
    Data preparing.
    '''
    groundTruth, ts_original = currentData[0], currentData[1]
    ts = [item for ind, item in enumerate(ts_original) if len(item) > 150 and len(item) < 160]
    index = [ind for ind, item in enumerate(ts_original) if len(item) > 150 and len(item) < 160]
    X_sc = MinMaxScaler()
    for ind in range(len(ts)):
        if max(ts[ind]) != min(ts[ind]):
            ts[ind] = (np.array(ts[ind]) - min(ts[ind]))/(max(ts[ind]) - min(ts[ind]))
    paddingTs = pad_sequences(ts, padding="pre", dtype="float64", value=-5)
    paddingTs[np.isnan(paddingTs)] = 0

    trainPrecent = 0.75
    train_index, valid_index = index[:int(trainPrecent * len(paddingTs))], index[int(trainPrecent * len(paddingTs)):]
    data_train, data_valid = paddingTs[:int(trainPrecent * len(paddingTs)), :], paddingTs[int(trainPrecent * len(paddingTs)):, :]
#    data_train, data_valid = data_train[:, :100], data_valid[:, :100]
    data_train_nums, data_valid_nums, ts_point_nums = len(data_train), len(data_valid), data_train.shape[1]
    
    tsTotalNums, tsTotalPtsNums = len(ts), len(data_train[0])
    
    data_train = data_train.reshape((data_train_nums, ts_point_nums, 1))
    data_valid = data_valid.reshape((data_valid_nums, ts_point_nums, 1))
    
    '''
    Training the seq2seq(GRU) model.
    '''
#    clf = seq2seq(layers=[200], input_length=tsTotalPtsNums,
#                  output_length=tsTotalPtsNums, nb_epoch=100, batch_size=512,
#                  input_dim=1, output_dim=1, predict_steps=1, early_stop_rounds=10)
#    clf.fit(data_train, data_valid, data_valid)
#    predicted_res = clf.predict(data_valid, num_steps_to_predict=ts_point_nums, batch_size=256)
    
    predicted_res = lstm_autoencoder(data_train, data_valid, time_steps=tsTotalPtsNums)
    
    plotInd = np.random.randint(0, len(data_valid), 40)
    plt.figure(figsize=(16, 9))
    for ind in plotInd:
        plt.plot(data_valid[ind][:, 0], color='b', linewidth=2)
        plt.plot(predicted_res[ind][:, 0], color='r', linewidth=2)
    plt.title("Validation Prediction")
#    
#    plotNoiseInd = np.random.choice([ind for ind, item in enumerate(valid_index) if groundTruth["label"].loc[item] == -1], 10)
#    plt.figure(figsize=(16, 9))
#    for ind in plotNoiseInd:
#        plt.plot(data_valid[ind][:, 0], color='b', linewidth=2)
#        plt.plot(predicted_res[ind][:, 0], color='r', linewidth=2)
#    plt.title("Noise Prediction")
        
#    # Prediction(Validating)
#    predicted_res = clf.predict(data_train, num_steps_to_predict=ts_point_nums, batch_size=256)
#    plotInd = np.random.randint(0, len(data_train), 20)
#    plt.figure()
#    for ind in plotInd:
#        plt.plot(data_train[ind], color='b', linewidth=2)
#        plt.plot(predicted_res[ind], color='r', linewidth=2)
#    plt.title("Training Prediction")