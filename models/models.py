import pandas as pd
import tensorflow as tf
import logging
from config.configs import *
import os
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelBase:
    def __init__(self):
        self.model_type = None
        self.train_data = None
        self.test_data = None
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(None, self.train_data.shape[1] - 1)),
            tf.keras.layers.Dropout(.20),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=5, mode='min', patience=4)
        self.model_size = None  # in gb
        self.columns = None
        self.scaler = MinMaxScaler()
        self.predictions = None

    def create_model(self, symbols):
        pass

    def train_test_model(self):
        pass

    def normalize_test(self):
        self.test_data.fillna(0)
        self.test_data = self.test_data._get_numeric_data()
        self.test_data = self.scaler.transform(self.test_data)

    def normalize_train(self):
        self.train_data.fillna(0)
        self.train_data = self.train_data._get_numeric_data()
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def batch_train(self):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1],
                                                              self.train_data[:, -1]))
        self.train_data = self.train_data.batch(8, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test(self):
        self.test_data = tf.data.Dataset.from_tensor_slices(
            (self.test_data[:, :-1],
             self.test_data[:, -1]))
        self.test_data = self.test_data.batch(1, drop_remainder=True).batch(1, drop_remainder=True)

    def batch(self, batch_train=False, batch_test=False):
        if batch_train: self.batch_train()
        if batch_test: self.batch_train()

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.fit(self.train_data, verbose=2, epochs=1, callbacks=self.early_stopping)

    def predict(self, data):
        data = tf.data.Dataset.from_tensor_slices((data[:, :-1]), data[:, -1])
        self.predictions = self.model.predict(data)

    def renormalize(self):
        x = list(self.train_data.as_numpy_iterator())  # do not renormalize any training data thats not in batches of 1
        #z = list(self.predictions.as_numpy_iterator())
        train_df = {}
        #prediction_df = {}

        for i in range(len(x)):
            values = x[i][0][0][0]
            train_df.update({i: values})
        #for i in range(len(z)):
        #    values = z[i][0][0][0]
        #    prediction_df.update({i: values})

        df_train = pd.DataFrame(train_df).transpose()
        renormalize_train = pd.DataFrame(self.train_data)

        #df_prediction = pd.DataFrame(prediction_df).transpose()
        #renormalize_prediction = pd.DataFrame(self.predictions)

        dataframe_to_inverse_transform_train = pd.concat([df_train, renormalize_train], axis=1, ignore_index=True)
        renormalized_training_values = self.scaler.inverse_transform(dataframe_to_inverse_transform_train)
        dataframe_to_inverse_transform_actual = pd.concat([df_prediction, renormalize_prediction], axis=1,
                                                          ignore_index=True)
        renormalized_predicted_values = self.scaler.inverse_transform(dataframe_to_inverse_transform_actual)

        self.df_train = pd.DataFrame(renormalized_training_values[:, :], columns=self.columns).rename(
            columns={'marketCap': 'marketCap_Predicted'})
        self.df_actual = pd.DataFrame(renormalized_predicted_values[:, :], columns=self.columns).rename(
            columns={'marketCap': 'marketCap_Predicted'})

        self.df_train['marketCap'] = self.close
        logger.info(f'{self.df_train.head(5)}')
        logger.info(f'{self.df_actual.head(5)}')

    def marketCap_to_sharePrice(self):
        df = pd.DataFrame()
        self.shares_outstanding = pd.read_sql('SELECT symbol,"sharesOutstanding" FROM stock_prices;',
                                              self.db_con).drop_duplicates()

        for symbol in self.shares_outstanding['symbol']:
            shares_outstanding = self.shares_outstanding.loc[self.shares_outstanding['symbol'] == symbol][
                'sharesOutstanding']
            temp = self.df_train.loc[self.df_train[symbol] == 1]
            temp_pred = self.df_actual.loc[self.df_actual[symbol] == 1]
            temp_pred['Price_Predicted'] = temp_pred['marketCap_Predicted'].apply(lambda x: x / shares_outstanding)
            temp['Price_Predicted'] = temp['marketCap_Predicted'].apply(lambda x: x / shares_outstanding)
            temp['Price_Actual'] = temp['marketCap'].apply(lambda x: x / shares_outstanding)
            df = pd.concat([df, temp, temp_pred])

        df['dateOffset'] = self.Date
        df.to_sql('test', self.db_con, if_exists='replace')


class SectorModel(ModelBase):
    pass

class SingleModel(ModelBase):
    def __init__(self, symbol):
        self.symbol = symbol
        self.train_data = pd.read_sql('SELECT * FROM market.model_data', con=POSTGRES_URL)



if __name__ == '__main__':
    Model = ModelBase()
    Model.train()
    Model.predict(data=None)
