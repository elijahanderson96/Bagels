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
    def __init__(self, train_data=None, test_data=None):
        self.model_type = None
        self.train_data = train_data
        self.test_data = test_data

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(None, self.train_data.shape[1] - 2)),
            tf.keras.layers.Dropout(.20),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=5, mode='min', patience=4)
        self.model_size = None  # in gb

        self.columns = None
        self.scaler = MinMaxScaler()
        self.predictions = None

    def normalize_test(self):
        self.test_data.fillna(0)
        self.test_data = self.test_data._get_numeric_data()
        self.test_data = self.scaler.transform(self.test_data)

    def normalize_train(self):
        self.train_data.fillna(0)
        self.train_data = self.train_data._get_numeric_data()
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def batch_train(self, batch_size=8):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1], self.train_data[:, -1]))
        self.train_data = self.train_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test(self, batch_size=1):
        self.test_data = tf.data.Dataset.from_tensor_slices((self.test_data[:, :-1], self.test_data[:, -1]))
        self.test_data = self.test_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def train(self):
        self.normalize_train()
        self.batch_train()
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.fit(self.train_data, verbose=2, epochs=1, callbacks=self.early_stopping)

    def predict(self):
        self.normalize_test()
        self.batch_test()
        self.predictions = self.model.predict(self.test_data)
        return self.renormalize_test_data()


    def renormalize_test_data(self):
        x = list(self.test_data.as_numpy_iterator())
        test_df = {}
        for i in range(len(x)):
            values = x[i][0][0][0]
            test_df.update({i: values})
        test_data = pd.DataFrame(test_df).transpose()  # predictions transposed
        predictions = pd.DataFrame(self.predictions)
        df_to_inverse_transform = pd.concat([test_data,predictions],axis=1,ignore_index=True)
        renormalized_df = pd.DataFrame(self.scaler.inverse_transform(df_to_inverse_transform), columns=self.columns)
        return renormalized_df


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
    def __init__(self):
        # right now its only from fundamentals but as we expand
        # our datasets, its likely we will need to source from more than just fundamentals
        self.train_data = pd.read_sql(f'SELECT * FROM market.model_data', con=POSTGRES_URL)
        self.test_data = self.train_data  # this is a hack for now.
        super().__init__(self.train_data, self.test_data)


if __name__ == '__main__':
    model = SingleModel()
