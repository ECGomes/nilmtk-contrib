from nilmtk.disaggregate import Disaggregator
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow._api.v1.keras.backend as K


class PB_Single(object):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        self.MODEL_NAME = 'PB_Single'
        self.window_size = params.get('window_size', 20)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.pb_value = params.get('pb_value', 0.5)
        self.n_features = params.get('n_features', 1)



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

        #TODO
        # 1 - Call preprocessing
        # 2 - Build model -> Kind of done
        # 3 - Train the model


        raise NotImplementedError()


    def disaggregate_chunk(self, test_mains):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()


    def call_preprocessing(self, train_mains, train_appliances):
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

        #TODO
        # 1 - Check for data formats
        # 2 - Implement for one variable, expand as needed

        print(train_mains)

        return train_mains, train_appliances


    def pb_loss(self):
        """Quantile loss function for model training"""

        def custom_loss(y_true, y_pred):
            tau = self.pb_value
            err = y_true - y_pred

            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)

        return custom_loss


    def return_network(self):
        """Returns the network for the PB-NILM single branch version"""

        #TODO
        # 1 - Check if loss function compiles
        # 2 - Check if model itself compiles

        inputs = tf.keras.layers.Input(shape=(self.window_size, self.n_features))

        # Define the network
        net = tf.keras.layers.Conv1D(128, kernel_size=5)(inputs)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        net = tf.keras.layers.MaxPooling1D()(net)

        net = tf.keras.layers.GRU(256)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        net = tf.keras.layers.Dense(1, activation='relu', name='Appliance')(net)

        model = tf.keras.Model(inputs=inputs, outputs=net)

        model.compile(optimizer='adam', loss=self.pb_loss)

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