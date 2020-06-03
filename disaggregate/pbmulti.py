from nilmtk.disaggregate import Disaggregator
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow._api.v1.keras.backend as K


# Adapted from the machinelearningmastery website:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def split_sequence(sequence, n_steps, n_intervals):
    X = list()
    for i in np.arange(0, len(sequence), n_intervals):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    return np.array(X)


class PB_Multi(Disaggregator):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        # Model Name
        self.MODEL_NAME = 'PB_Multi'
        self.models = {}

        # Network inputs
        self.window_size = params.get('window_size', 20)
        self.n_features = params.get('n_features', 1)

        # Training parameters
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.pb_value = params.get('pb_value', 0.5)

        # Dropout parameters
        self.use_dropout = params.get('use_dropout', True)
        self.dropout_rate = params.get('dropout_rate', 0.2)

        # MaxPooling1D options
        self.use_maxpool = params.get('use_maxpool', False)

    def partial_fit(self, train_mains, train_appliances, **load_kwargs):
        """ Trains the model given a metergroup containing appliance meters
        (supervised) or a site meter (unsupervised).  Will have a
        default implementation in super class.
        train_main: list of pd.DataFrames with pd.DatetimeIndex as index and 1
                    or more power columns
        train_appliances: list of (appliance_name,list of pd.DataFrames) with
                        the same pd.DatetimeIndex as index as train_main and
                        the same 1 or more power columns as train_main
        """

        processed_x, processed_y = self.preprocessing_xy(train_mains, train_appliances)

        for appliance in processed_x.keys():
            x_train, x_test, \
            y_train1, y_test1, \
            y_train2, y_test2, \
            y_train3, y_test3 = train_test_split(processed_x[appliance],
                                                 processed_y[appliance][0],
                                                 processed_y[appliance][1],
                                                 processed_y[appliance][2],
                                                 test_size=0.15)

            network = self.return_network()

            # Validation split set to 0.1765 to have roughly 70/15/15 set sizes
            network.fit(x_train, [y_train1, y_train2, y_train3],
                        epochs=self.n_epochs,
                        batch_size=self.batch_size,
                        verbose=1,
                        validation_split=0.1765)

            self.models[appliance] = network

    def disaggregate_chunk(self, test_mains, appliance_only=True):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        appliance_only: return only the appliance branch of the network
        """

        test_predictions = []
        preprocessed_x = self.preprocess_x(test_mains)

        disaggregated_dict = {}

        for appliance in np.arange(len(preprocessed_x)):
            current_model = list(self.models.keys())[appliance]
            current_appliance = list(preprocessed_x.keys())[appliance]

            temp_model = self.models[current_model]

            ypred = temp_model.predict(preprocessed_x[current_appliance],
                                       verbose=1,
                                       batch_size=self.batch_size)

            if appliance_only:
                ypred = ypred[0].reshape(-1, 1).ravel()
            else:
                return ypred

            # Prepend zeros to the predictions
            temp_nans = [0 for i in np.arange(self.window_size)]
            temp_nans = np.array(temp_nans)

            temp_series1 = pd.Series(temp_nans)
            temp_series2 = pd.Series(ypred)

            temp_series3 = pd.concat([temp_series1, temp_series2])

            disaggregated_dict[current_model] = temp_series3

        test_predictions.append(pd.DataFrame(disaggregated_dict))

        return test_predictions

    def preprocess_x(self, test_mains):

        dict_mains = {}

        for appliance in np.arange(len(test_mains)):
            temp_x_list = []

            for feature in test_mains[appliance]['power'].keys():
                split_x = split_sequence(test_mains[appliance]['power'][feature], self.window_size, 1)
                temp_x_list.append(split_x)

            temp_x = np.stack(temp_x_list, axis=-1)
            dict_mains['appliance{:02d}'.format(appliance)] = temp_x

        return dict_mains

    def preprocessing_xy(self, train_mains, train_appliances):
        """Calls the preprocessing functions of this algorithm and returns the
           preprocessed data in the same format
        Parameters
        ----------
        train_main: list of pd.DataFrames with pd.DatetimeIndex as index and 1
                    or more power columns
        train_appliances: list of (appliance_name,list of pd.DataFrames) with
                    the same pd.DatetimeIndex as index as train_main and the
                    same 1 or more power columns as train_main
        """

        # train_appliances is a tuple:
        # train_appliances[0][0] is the name of the appliance
        # train_appliances[0][1] is the first features of the appliance

        print('Preprocessing...')

        dict_mains = {}
        dict_appliances = {}

        for appliance in np.arange(len(train_appliances)):
            temp_x_list = []

            for feature in train_mains[appliance]['power'].keys():
                split_x = split_sequence(train_mains[appliance]['power'][feature], self.window_size, 1)
                temp_x_list.append(split_x)

            temp_x = np.stack(temp_x_list, axis=-1)
            temp_y1 = train_appliances[appliance][1][0]['power'].iloc[self.window_size:].values
            temp_y3 = train_mains[appliance]['power'][feature].iloc[self.window_size:].values
            temp_y3 = np.reshape(temp_y3, (temp_y3.shape[0], 1))

            temp_y2 = temp_y3 - temp_y1

            dict_mains[train_appliances[appliance][0]] = temp_x
            dict_appliances[train_appliances[appliance][0]] = [temp_y1, temp_y2, temp_y3]

        print('Finished preprocessing...')

        return dict_mains, dict_appliances

    def pb_loss(self):
        """Quantile loss function for model training"""

        def custom_loss(y_true, y_pred):
            tau = self.pb_value
            err = y_true - y_pred

            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)

        return custom_loss

    def return_network(self):
        """
        Returns the network for the PB-NILM single branch version from the paper:
        https://ieeexplore.ieee.org/document/9025262
        """

        def pb_loss(y_true, y_pred):
            tau = self.pb_value
            err = y_true - y_pred

            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)

        inputs = tf.keras.layers.Input(shape=(self.window_size, self.n_features))

        # Define the first branch of the network
        branch1 = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        branch1 = tf.keras.layers.BatchNormalization()(branch1)
        branch1 = tf.keras.layers.Activation('relu')(branch1)

        if self.use_maxpool:
            branch1 = tf.keras.layers.MaxPooling1D()(branch1)

        branch1 = tf.keras.layers.GRU(256)(branch1)
        branch1 = tf.keras.layers.BatchNormalization()(branch1)
        branch1 = tf.keras.layers.Activation('relu')(branch1)

        if self.use_dropout:
            branch1 = tf.keras.layers.Dropout(rate=self.dropout_rate)(branch1)

        branch1 = tf.keras.layers.Dense(1, activation='relu', name='Appliance')(branch1)


        # Define the first branch of the network
        branch2 = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        branch2 = tf.keras.layers.BatchNormalization()(branch2)
        branch2 = tf.keras.layers.Activation('relu')(branch2)

        if self.use_maxpool:
            branch2 = tf.keras.layers.MaxPooling1D()(branch2)

        branch2 = tf.keras.layers.GRU(256)(branch2)
        branch2 = tf.keras.layers.BatchNormalization()(branch2)
        branch2 = tf.keras.layers.Activation('relu')(branch2)

        if self.use_dropout:
            branch2 = tf.keras.layers.Dropout(rate=self.dropout_rate)(branch2)

        branch2 = tf.keras.layers.Dense(1, activation='relu', name='Diff')(branch2)

        branch3 = tf.keras.layers.Add(name='Total')([branch1, branch2])

        model = tf.keras.Model(inputs=inputs, outputs=[branch1, branch2, branch3])

        model.compile(optimizer='adam', loss=pb_loss)

        return model

    def save_model(self, folder_name):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()

    def load_model(self, folder_name):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()
