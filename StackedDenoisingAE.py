# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:25:52 2019

@author: michaelyin1994
"""
from datetime import datetime
import numpy as np
import scipy.sparse as scp
import seaborn as sns
#from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import MinMaxScaler

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def batch_generator(X=None, Y=None, batch_size=128, shuffle=True, seed=2019):
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
            x_batch = X[batch_index, :].toarray()
            y_batch = Y[batch_index, :].toarray()
        else:
            x_batch = X[batch_index, :]
            y_batch = Y[batch_index, :]
        counter += 1

        # Return the generator
        yield x_batch, y_batch

        # Reset the count variable, which means the same batch will
        # be generated in the next epoch.
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def x_generator(X=None, batch_size=128, shuffle=True, seed=2019):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sample_index)
    
    sparse = False
    if scp.issparse(X):
        sparse = True
        
    while counter < number_of_batches: 
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        if sparse:
            x_batch = X[batch_index, :].toarray()
        else:
            x_batch = X[batch_index, :]
        yield x_batch, batch_index
        counter += 1


def kullback_leibler_divergence(y_true, y_pred):
    '''
    @Description:
    ----------
    Implement the KL divergence.

    @Parameters:
    ----------

    @Return:
    ----------
    None.
    '''
    
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


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


class StackedDenoisingAE(object):
    '''
    ----------
    Author: Michael Yin
    Date: 2019/08/01
    Modified: 2019/09/10
    Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Constructing the Stacked Denoising Autoencoder(SDA) based on the paper [2] and [3]. The source code is
    constructed by [1], and modified by Michael Yin(zhuoyin94@163.com).
    
    @Parameters:
    ----------
    self.n_layers: int-like
        The number of hidden layers of the SDA.

    self.n_hid: list-like
        The number of hidden units of each hidden layer of SDA.

    self.dropout: list-like
        The dropout rate of inputs for each decoding-encoding network.
    
    self.regularizer_l1: list-like
        pass
    
    self.regularizer_l2: list-like
        pass    
    
    self.enc_act: list-like
        The activation function of each encoding layer.

    self.dec_act: list-like
        The activation function of each decoding layer.

    self.bias: bool-like
        Whether each hidden layer contains a bias vector.

    self.batch_size, self.nb_epoch: int-like
        Batch size and the number of epoches used to train the decoding-encoding network.

    self.optimizer: str-like
        Optimizer used to train the neural network.

    @Methods:
    ----------
    self.get_pretrained_sda(self, data_in, data_val, data_test, dir_out="..//Models//",
                            get_enc_model=True, write_model=True, model_layers=None):
        Training the Stacked Denoising Autoencoder(SDA). If the path is specified, then load the pre-trained
        model from disk.

    self.supervised_classification(self, model, x_train=[], x_val=[], x_test=[], y_train=[], y_val=[], y_test=[],
                                   n_classes=2, final_act_fn='softmax', loss='categorical_crossentropy', get_recon_error=False):
        Applying the pre-trained weights in a softmax classifier to construct a fine-tuned network.

    self.evaluate_on_test(self, fit_model, x_test, y_test, n_classes):
        Evaluating the test data.

    self._get_nth_layer_output(self, model, n_layer, X, train=1):
        Get the n-th layer output of the model. The parameter "train" determines the drop-out behavior.
        If train == 1, it means the model is in training mode, and the model needs the drop-out layer;
        If train == 0, it means the model is in testing model, and the model DO NOT need the drop-out layer.

    self._get_intermediate_output(self, model=None, data_in=None, n_layer=None, train=None,
                                 n_out=None, batch_size=128, dtype=np.float32):
        Get the intermediate output of the neural network. It is a wrapper function of self._get_nth_layer_output.

    self._build_model_from_encoders(self, encoding_layers, dropout_all=False):
        This function rebuild a neural network based on the encoding_layers. encoding_layers is a list that each
        element in the list is a pre-trianed hidden layer. Note that the first layer of the rebuild network is
        DROP-OUT layer. This rebuild function is mainly used for the supervised retraining.

    self._assert_input(self, n_layers, n_hid, dropout, enc_act, dec_act):
        The santiy check for the input parameters.

    self._get_recon_error(self, model=None, data_in=None, n_out=None):
        Get the reconstruct error between the model outputs and the data_in.

    @References:
    ----------
    [1] https://github.com/MadhumitaSushil/SDAE/blob/master/sdae.py
    [2] Lu, Chen, et al. "Fault diagnosis of rotary machinery components using a stacked denoising autoencoder-based health state identification." Signal Processing 130 (2017): 377-388.
    [3] Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." Journal of machine learning research 11.Dec (2010): 3371-3408.
    [4] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.
    '''
    def __init__(self, n_layers=1, n_hid=[500], dropout=[0.05],
                 enc_act=['sigmoid'], dec_act=['linear'],
                 weight_regularizer=[], activity_regularizer=[],
                 bias=True, loss_fn='mse', optimizer='rmsprop',
                 batch_size=32, early_stop_rounds=10, nb_epoch=300, verbose=1):
        
        self.n_layers = n_layers
        self.n_hid, self.dropout, self.enc_act, self.dec_act = self._assert_input(n_layers,
                                                                                  n_hid, dropout,
                                                                                  enc_act, dec_act)
        self.weight_regularizer, self.activity_regularizer = self._regularizer_check(n_layers,
                                                                                     weight_regularizer,
                                                                                     activity_regularizer)
        self.bias = bias
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.verbose = verbose
        self.early_stop_rounds = early_stop_rounds
        
        
    def get_pretrained_sda(self, data_in, data_val, data_test, dir_out=".//Models//",
                           get_enc_model=True, write_model=True, model_layers=None):
        '''
        @Description:
        ----------
        -- Pretrains layers of a stacked denoising autoencoder to generate
        low-dimensional representation of data.
        -- Returns a Sequential model with the Dropout layer and pretrained
        encoding layers added sequentially.
        -- Optionally, we can return a list of pretrained sdae models by
        setting get_enc_model to False.
        -- Additionally, returns dense representation of input, validation and test data.
        This dense representation is the value of the hidden node of the last layer.
        -- The cur_model be used in supervised task by adding a classification/regression layer on top,
        or the dense pretrained data can be used as input of another cur_model.

        @Parameters:
        ----------
        data_in, data_val, data_test: {N * M} numpy-array
            The data used to train/valiadate/test. The shape should be N * M, where N is the 
            number of training/validating/testing data, and M is the number of features.
        
        dir_out: str-like
            The path that the key parameters of the stacked autoencoder outputs.
        
        get_enc_model: bool-like
            If get_enc_model is True, return NN with dropout layer, else just return the 
            trained layer: encoders.
        
        write_model: bool-like
            Whether to write the model to the local path.
        
        model_layers: bool-like
            If True, return pretrained cur_model layers, to continue training pretrained model_layers, if required.
        
        @BugToBeFixed:
        ----------
        1. self._build_model_from_encoders rebuild the model with the dropout layer, why ? ? ? ?
        '''
        if model_layers is not None:
            self.n_layers = len(model_layers)
        else:
            model_layers = [None] * self.n_layers
        
        encoders, recon_mse = [], []
        print("\nStart hidden layer pretraining:")
        print("=================================================================")
        print("@Start at {}".format(datetime.now()))
        for cur_layer in range(self.n_layers):
            
            if model_layers[cur_layer] is None:
                
                # Create the input layer
                input_layer = Input(shape=(data_in.shape[1], ))
                
                # Create the dropout layer
                dropout_layer = Dropout(self.dropout[cur_layer])
                in_dropout = dropout_layer(input_layer)
                
                # Create the encodoing layer
                encoder_layer = Dense(output_dim=self.n_hid[cur_layer],
                                      init='glorot_uniform',
                                      activation=self.enc_act[cur_layer],
                                      name='encoder_'+str(cur_layer), bias=self.bias,
                                      kernel_regularizer=self.weight_regularizer[cur_layer],
                                      activity_regularizer=self.activity_regularizer[cur_layer])
                encoder = encoder_layer(in_dropout)
                
                # Create the decoding layer
                n_out = data_in.shape[1]
                decoder_layer = Dense(output_dim=n_out, bias=self.bias,
                                      init='glorot_uniform',
                                      activation=self.dec_act[cur_layer],
                                      name='decoder_'+str(cur_layer))
                decoder = decoder_layer(encoder)
                
                # Complie the model
                cur_model = Model(input_layer, decoder)
                cur_model.compile(loss=self.loss_fn, optimizer=self.optimizer)
                cur_model.summary()
            else:
                cur_model = model_layers[cur_layer]
                cur_model.summary()

            # Start fitting the Autoencoder
            print("Current training layer {}.".format(str(cur_layer)))
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=self.early_stop_rounds,
                                           verbose=0)
            hist = cur_model.fit_generator(generator=batch_generator(data_in, data_in,
                                                                     batch_size=self.batch_size,
                                                                     shuffle=True),
            callbacks=[early_stopping],
            nb_epoch=self.nb_epoch,
            samples_per_epoch=np.ceil(data_in.shape[0]/self.batch_size),
            verbose=self.verbose,
            validation_data=batch_generator(data_val, data_val,
                                            batch_size=self.batch_size,
                                            shuffle=False),
                                            nb_val_samples=np.ceil(data_val.shape[0]/self.batch_size))
            print("Layer " + str(cur_layer) + " has been trained.")
            
            # -----------------------------------------
            # Step 1: Saving the trained layder to the model_layers
            # Step 2: Saving the encoding layer to the encoders
            model_layers[cur_layer] = cur_model
            encoder_layer = cur_model.layers[-2]
            encoders.append(encoder_layer)
            # -----------------------------------------

            # Save the recovery error information
            recon_mse.append(self._get_recon_error(cur_model, data_in,
                                                   n_out=cur_model.layers[-1].output_shape[1]))
            
            # -----------------------------------------
            # Step 1: GET THE HIDDEN OUTPUT AS THE TRAINING DATA.(WITHOUT DROPOUT)
            # Regenerating the input data, train == 0 means we do not want to use
            # dropout to get hidden node value, since is a train-only behavior, 
            # but here we want to get the hidden layer output of the data_in.

            # Layer (type)                 Output Shape              Param #
            # =================================================================
            # input_1 (InputLayer)         (None, 784)               0
            # _________________________________________________________________
            # dropout_1 (Dropout)          (None, 784)               0
            # _________________________________________________________________
            # encoder_0 (Dense)            (None, 400)               314000    (Layer need to be outputed, n_layer=2)
            # _________________________________________________________________
            # decoder_0 (Dense)            (None, 784)               314384
            # =================================================================
            data_in = self._get_intermediate_output(cur_model, data_in, n_layer=2,
                                                    train=0, n_out=self.n_hid[cur_layer],
                                                    batch_size=self.batch_size)
            
            # Step 2: GET THE HIDDEN OUTPUT AS THE VALIDATION DATA AND TESTING DATA.(WITHOUT DROPOUT)
            data_val = self._get_intermediate_output(cur_model, data_val,
                                                     n_layer=2, train=0,
                                                     n_out=self.n_hid[cur_layer],
                                                     batch_size=self.batch_size)
            data_test = self._get_intermediate_output(cur_model, data_test,
                                                      n_layer=2, train=0,
                                                      n_out=self.n_hid[cur_layer],
                                                      batch_size=self.batch_size)
            # -----------------------------------------
            
            # Normalization
            X_sc = MinMaxScaler()
            data_in = X_sc.fit_transform(data_in)
            data_val = X_sc.fit_transform(data_val)
            data_test = X_sc.fit_transform(data_test)
            
        # Write down some important parameters information in **.txt form.
        self._write_sda_config(dir_out)
        
        # If get_enc_model == True: Buliding a NN with dropout, the output is the high-level representations.
        if get_enc_model:
            final_model = self._build_model_from_encoders(encoders, dropout_all=False)#, final_act_fn= final_act_fn)
            if write_model:
                save_model(final_model, out_dir=dir_out, f_arch='enc_layers.png',
                           f_model='enc_layers.json', f_weights='enc_layers_weights.h5')
        else:
            final_model = model_layers
        print("@End pre-training model at {}".format(datetime.now()))
        print("=================================================================")
        return final_model, (data_in, data_val, data_test), recon_mse


    def supervised_classification(self, model, x_train=[], x_val=[], x_test=[], y_train=[], y_val=[], y_test=[],
                                  n_classes=2, final_act_fn='softmax', loss='categorical_crossentropy', get_recon_error=False):
        '''
        @Description:
        ----------
        Supervised learning based on the trained stacked autoencoder(model).

        @Parameters:
        ----------
        model: Keras-object-like
            Pretrained model or preloaded model from self.get_pretrained_sda() method.

        x_train, x_val, x_test, y_train, y_val, y_test: numpy-array-like
            Training/validating/testing data and their corresponding labels.
        
        n_classes: numpy-array-like
            The numbe of the classes.
        
        get_enc_model: bool-like
            pass
        
        final_act_fn: bool-like
            pass
        
        loss: bool-like
            pass
        
        get_recon_error:
            pass

        @BugToBeFixed:
        ----------
        1. The model is a Stacked Denoising Autoencoder(SDA), which is trained by the self.get_pretrained_sda method,
           which means the model already has the output layer, and the shape is hidden_last * number_of_outputs. When
           we directly add a new Dense layer at the end of the pre-trained network, the topology will have 2 output nodes.
           This means method self._get_nth_layer_output will have the indexing error.
        '''

        print("Training fine-tuned neural network:")
        print("=================================================================")
        print("Start fine-tuned network training model at {}".format(datetime.now()))
        # Rebuild the model using the pre-trained weights
        model.add(Dense(n_classes, activation=final_act_fn))
        model.compile(loss=loss, optimizer=self.optimizer)
        
        # Early stopping to stop training when val loss increses for 2 epoch
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
        hist = model.fit_generator(generator = batch_generator(x_train, y_train,
                                                               batch_size=self.batch_size,
                                                               shuffle=True),
                                                               samples_per_epoch=x_train.shape[0],
                                                               callbacks=[early_stopping],
                                                               nb_epoch=self.nb_epoch,
                                                               verbose=self.verbose,
                                                               validation_data=batch_generator(x_val, y_val,
                                                                                               batch_size = self.batch_size,
                                                                                               shuffle = False),
                                                               nb_val_samples = x_val.shape[0])
        '''
        # Get the final hidden layer output of fine-tuned SDAE
        final_train = self._get_intermediate_output(model, x_train, n_layer=-2, train=0,
                                                    n_out=model.layers[-2].output_shape[1],
                                                    batch_size=self.batch_size)
        final_val = self._get_intermediate_output(model, x_val, n_layer=-2, train=0,
                                                  n_out=model.layers[-2].output_shape[1],
                                                  batch_size=self.batch_size)
        
        if len(x_test) != 0:
            final_test = self._get_intermediate_output(model, x_test, n_layer=-2,
                                                       train=0, n_out=model.layers[-2].output_shape[1],
                                                       batch_size=self.batch_size)
        else:
            final_test = None
           
        # Get reconstruction error of final nodes
        if get_recon_error:
            recon_mse = self._get_recon_error(model, x_train, n_out=n_classes)
        else:
            recon_mse = None
        print("End fine-tuned network training model at {}".format(datetime.now()))
        print("=================================================================")
        return model, (final_train, final_val, final_test), recon_mse
        '''
        print("End fine-tuned network training model at {}".format(datetime.now()))
        print("=================================================================")
        return model


    def evaluate_on_test(self, fit_model, x_test, y_test, n_classes):
        """
        Evaluate a trained model on test dataset 
        Use this function only for the final evaluation, not for development
        """
        fit_model.evaluate_generator(generator = batch_generator(x_test, y_test,
                                                                 batch_size = self.batch_size,
                                                                 shuffle = False),
                                                                 samples = x_test.shape[0],)


    def _write_sda_config(self, dir_out):
        """
        Write the configuration of the autoencoder to a file
        @param cur_sdae: autoencoder class
        @param cfg: config object
        """
        with open(dir_out + 'sdae_config.txt', 'w') as f:
            f.write("Number of layers: " + str(self.n_layers))
            f.write("\nHidden nodes: ")
            for i in range(self.n_layers):
                f.write(str(self.n_hid[i]) + ' ')
                
            f.write("\nDropout: ")
            for i in range(self.n_layers):
                f.write(str(self.dropout[i]) + ' ')
            
            f.write("\nEncoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.enc_act[i]) + ' ')
                
            f.write("\nDecoder activation: ")
            for i in range(self.n_layers):
                f.write(str(self.dec_act[i]) + ' ')
            
            f.write("\nEpochs: " + str(self.nb_epoch))
            
            f.write("\nBias: " + str(self.bias))
            f.write("\nLoss: " + str(self.loss_fn))
            f.write("\nBatch size: " + str(self.batch_size))
            f.write("\nOptimizer: " + str(self.optimizer))


    def _get_intermediate_output(self, model=None, data_in=None, n_layer=None, train=None,
                                 n_out=None, batch_size=128, dtype=np.float32):
        '''
        @Description:
        ----------
        Get the intermediate output of the neural network.

        @Parameters:
        ----------
        model: keras model-like
            Model to get output from.

        data_in: numpy array-like
            Sparse representation of input data.

        n_layer: int-like
            The layer number for which output is required.

        train: int-like
            1 to use the same setting as TRAINING (for example, with Dropout, etc.), 0 to use the same setting as TESTING phase for the model.

        n_out: int-like
            The number of units of HIDDEN OUPUTS.

        batch_size: int-like
            The num of instances to convert to dense at a time.
        @Returns:
        ----------
        Return the intermediate output of the neural network.
        '''
        data_out = np.zeros(shape = (data_in.shape[0], n_out))
        
        x_batch_gen = x_generator(data_in, batch_size=batch_size, shuffle=False)
        stop_iter = int(np.ceil(data_in.shape[0] / batch_size))
        
        for i in range(stop_iter):
            cur_batch, cur_batch_idx = next(x_batch_gen)
            data_out[cur_batch_idx, :] = self._get_nth_layer_output(model, n_layer, X=cur_batch, train=train)
        
        return data_out.astype(dtype, copy = False)
            
    
    def _get_nth_layer_output(self, model, n_layer, X, train=1):
        '''
        @Description:
        ----------
        THE CORE FUNCTION. Using Keras backend function generating the K.function object.
        K.function([Tensorflow Input Placeholder Object, Learning Phase], [Tensorflow Output Placeholder Object](Need to specify the output layer))
        K.function will generate a instance.

        @Parameters:
        ----------
        model: keras model-like
            Keras model to get an intermediate value out of.

        n_layer: int-like
            The layer number to get the value of.

        X: numpy array-like
            Input data for which layer value should be computed and returned.

        train: int-like
            1 to use the same setting as TRAINING (for example, with Dropout, etc.), 0 to use the same setting as TESTING phase for the model.

        @Returns:
        ----------
        Return the value of n_layer in the given model, input, and setting
        '''
        get_nth_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[n_layer].output])
        return get_nth_layer_output([X, train])[0]


    def _get_recon_error(self, model=None, data_in=None, n_out=None):
        '''
        @Description:
        ----------
        Return reconstruction squared error at individual nodes, averaged across all instances.
        Here, train = 0 because we do not want to use dropout to get hidden node value,
        since is a train-only behavior, used only to learn weights. output of third layer: output layer

        @Parameters:
        ----------
        model: keras model-like
            trained model

        data_in: list-like
            Input data to reconstruct

        n_out: list-like
            The number of model output nodes

        @Returns:
        ----------
        The reconstruction error.(int-like)
        '''
        train_recon = self._get_intermediate_output(model, data_in, n_layer=-1, train=0,
                                                    n_out=n_out, batch_size=self.batch_size) 
        recon_mse = np.mean(np.square(train_recon - data_in), axis=0)
        
        recon_mse = np.ravel(recon_mse)
        mean_recon_mse = np.mean(recon_mse)
        return mean_recon_mse


    def _assert_input(self, n_layers, n_hid, dropout, enc_act, dec_act):
        '''
        @Description:
        ----------
        If the hidden nodes, dropout proportion, encoder activation function
        or decoder activation function is given, it uses the same parameter
        for all the layers.
        Errors occurs when there is a size dismatch between the number of
        layers and parameters of each layer.

        @Parameters:
        ----------
        n_layers: int-like
            The number of hidden layer of the Stacked Denoising Autoencoder.
        
        dropout: list-like
            The number of hidden units of each decoding-encoding network.
        
        enc_act: list-like
            The activation function of each encoding layer.
        
        dec_act: list-like
            The activation function of each decoding layer.

        @Returns:
        ----------
        Input parameters that modified by the sanity check.
        e.g.:
        dropout == [0.01] where n_hid == 3, then return dropout == [0.01, 0.01, 0.01]
        '''
        if isinstance(n_layers, int):
            pass
        else:
            raise TypeError("Unkown n_layers Parameter!")
        
        if isinstance(n_hid, list) and isinstance(dropout, list) and isinstance(enc_act, list) and isinstance(dec_act, list):
            pass
        else:
            raise TypeError("Unknow parameters !")
        
        if len(n_hid) == 1:
            n_hid = n_hid * n_layers
        
        # Expand the dropout list
        if len(dropout) == 1:
            dropout = dropout * n_layers
        
        for i in dropout:
            assert (i <= 1 and i >= 0), "The dropout rate is not correct !"
        
        if len(enc_act) == 1:
            enc_act = enc_act * n_layers
        
        if len(dec_act) == 1:
            dec_act = dec_act * n_layers
                
        assert (n_layers == len(n_hid) == len(dropout) == len(enc_act) == len(dec_act)), "Please specify as many hidden nodes, dropout proportion on input, and encoder and decoder activation function, as many layers are there, using list data structure"
        return n_hid, dropout, enc_act, dec_act
    
    
    def _regularizer_check(self, n_layers, weight_regularizer, activity_regularizer):
        '''
        @Description:
        ----------
        Check the regularizer parameter for each layer.
        
        @Parameters:
        ----------
        n_layers: int-like
            The number of hidden layer of the Stacked Denoising Autoencoder.
        
        weight_regularizer: list-like
            The number of hidden units of each decoding-encoding network.
        
        activity_regularizer: list-like
            The activation function of each encoding layer.
        
        @Returns:
        ----------
        
        '''
        if isinstance(weight_regularizer, list) and isinstance(activity_regularizer, list):
            pass
        else:
            raise TypeError("Invalid regularizer !")
        
        # weight_regularizer check.
        if len(weight_regularizer) == 0:
            weight_regularizer = [None] * n_layers
            
        if len(weight_regularizer) != n_layers and len(weight_regularizer) == 1:
            weight_regularizer = weight_regularizer * n_layers
            
        if len(weight_regularizer) != n_layers:
            raise ValueError("n_layers != the length of weight_regularizer !")
        
        # activity_regularizer check.
        if len(activity_regularizer) == 0:
            activity_regularizer = [None] * n_layers
            
        if len(activity_regularizer) != n_layers and len(activity_regularizer) == 1:
            activity_regularizer = activity_regularizer * n_layers
            
        if len(activity_regularizer) != n_layers:
            raise ValueError("n_layers != the length of activity_regularizer !")     
        return weight_regularizer, activity_regularizer
    
    
    def _build_model_from_encoders(self, encoding_layers, dropout_all=False):
        '''
        @Description:
        ----------
        Builds a deep NN model that generates low-dimensional representation of input, based on pretrained layers.

        @Parameters:
        ----------
        encoding_layers: list-like
            Pretrained encoder layers. It is a list that each element in the list is the Keras object().
        
        dropout_all: bool-like
            If False, then there is no dropout layer between the hidden and hidden layer. But the first
            layer of the hidden layer is the dropout layer, with the dropout rate self.dropout[0].
        
        @Returns:
        ----------
        Keras model object.
        '''
        model = Sequential()
        model.add(Dropout(self.dropout[0], input_shape = (encoding_layers[0].input_shape[1], )))
        
        for i in range(len(encoding_layers)):
            if i and dropout_all:
                model.add(Dropout(self.dropout[i]))
                
            encoding_layers[i].inbound_nodes = []
            model.add(encoding_layers[i])
        return model
