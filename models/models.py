import pandas as pd
import tensorflow as tf
import logging
from config.configs import *
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
        self.train_dates = None
        self.test_dates = None
        self.symbol = None
        self.sector = None
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=5, mode='min', patience=4)
        self.model_size = None  # in gb
        self.columns = None
        self.scaler = MinMaxScaler()
        self.predictions = None
        self.trained = False

    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(None, self.train_data.shape[1] - 1)),
            tf.keras.layers.Dropout(.20),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)])

    def normalize_test(self):
        self.test_data.fillna(0)
        self.test_dates = self.test_data.pop('date')
        self.test_data['marketcap'] = [1] * len(self.test_data)
        self.test_data = self.test_data._get_numeric_data()
        self.test_data = self.scaler.transform(self.test_data)

    def normalize_train(self):
        self.train_data.fillna(0)
        self.train_dates = self.train_data.pop('date')
        self.train_data = self.train_data._get_numeric_data()
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def batch_train(self, batch_size=8):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1], self.train_data[:, -1]))
        self.train_data = self.train_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test(self, batch_size=1):
        self.test_data = tf.data.Dataset.from_tensor_slices((self.test_data[:, :-1], self.test_data[:, -1]))
        self.test_data = self.test_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def one_hot_encode(self):
        self.train_data = pd.get_dummies(self.train_data, columns=['symbol'])
        self.test_data = pd.get_dummies(self.test_data, columns=['symbol'])

    def train(self):
        if self.sector: self.one_hot_encode()  # dont need to one hot encode single models
        self.normalize_train()
        self.create_model()
        self.batch_train()
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.fit(self.train_data, verbose=2, epochs=1, callbacks=self.early_stopping)
        self.trained = True

    def predict(self):
        assert self.trained, 'Must train model before attempting prediction'
        self.normalize_test()
        self.batch_test()
        self.predictions = self.model.predict(self.test_data)
        self.post_process()

    def post_process(self):
        self._renormalize_test_data()
        self._finalize_predictions()

    def _finalize_predictions(self):
        self.predictions['date'] = self.test_dates.to_list()
        symbol_or_sector = 'symbol' if self.symbol else 'sector'
        if self.symbol: self.predictions[symbol_or_sector] = self.symbol
        self._market_cap_to_share_price()

    def _renormalize_test_data(self):
        x = list(self.test_data.as_numpy_iterator())
        test_df = {}
        for i in range(len(x)):
            values = x[i][0][0][0]
            test_df.update({i: values})
        test_data = pd.DataFrame(test_df).transpose()  # predictions transposed
        predictions = pd.DataFrame(self.predictions)
        df_to_inverse_transform = pd.concat([test_data, predictions], axis=1, ignore_index=True)
        self.predictions = pd.DataFrame(self.scaler.inverse_transform(df_to_inverse_transform), columns=self.columns)

    def _market_cap_to_share_price(self):
        if self.symbol: self.predictions = self.predictions[['symbol', 'date', 'marketcap']]
        # symbols = self.predictions
        # shares_outstanding = int(pd.read_sql(f'SELECT * '
        #                                     f'FROM market.stock_prices '
        #                                     f'WHERE symbol in {tuple(self.predictions["symbol"].unique())}',
        #                                     con=POSTGRES_URL).squeeze())


class SectorModel(ModelBase):
    def __init__(self, sector):
        super().__init__()
        self.sector = sector

    def train(self):
        self.gen_feature_matrix()
        super().train()

    def predict(self):
        super().predict()

    def gen_feature_matrix(self):
        logger.info(f'Fetching current data for {self.symbol}')
        self._fetch_data()

    def _fetch_data(self):
        self.data = pd.read_sql(f'SELECT * FROM market.fundamentals_interpolated '
                                f"WHERE symbol in ('A','AA') "  # TODO MAKE DYNAMIC
                                f'ORDER BY date ASC; ', con=POSTGRES_URL)
        self.data['date'] = self.data['date'].astype('datetime64')
        self._train_test_split()
        self._assign_labels()

    def _assign_labels(self):
        logger.info(f'Assigning labels to the dataset')
        labels = pd.read_sql(f'SELECT date, close * shares_outstanding as marketcap, symbol '
                             f'FROM market.stock_prices '
                             f"WHERE symbol in ('A','AA')", con=POSTGRES_URL)
        self.train_data = pd.merge(left=self.train_data, right=labels, on=['date', 'symbol'])
        logger.info(f'Training data is now of shape {self.train_data.shape} for {self.symbol} after assigning labels')

    def _train_test_split(self):
        logger.info(f'Whole dataset is of shape {self.data.shape} for {self.symbol}')
        logger.info(f'Splitting into train and test sets')
        self.train_data = self.data.loc[self.data['date'] < datetime.now()]
        self.test_data = self.data.loc[self.data['date'] >= datetime.now()]
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Testing data is of shape {self.test_data.shape}')


class SingleModel(ModelBase):
    def __init__(self, symbol):
        # right now its only from fundamentals but as we expand
        # our datasets, its likely we will need to source from more than just fundamentals
        super().__init__()
        self.symbol = symbol.upper()
        self.train_data = None
        self.test_data = None
        self.dates = None

    def train(self):
        self.gen_feature_matrix()
        super().train()

    def predict(self):
        super().predict()

    def gen_feature_matrix(self):
        logger.info(f'Fetching current data for {self.symbol}')
        self._fetch_data()

    def _fetch_data(self):
        self.data = pd.read_sql(f'SELECT * FROM market.fundamentals_interpolated '
                                f"WHERE symbol='{self.symbol}' "
                                f'ORDER BY date ASC; ', con=POSTGRES_URL)
        self.data['date'] = self.data['date'].astype('datetime64')
        self._train_test_split()
        self._assign_labels()

    def _assign_labels(self):
        logger.info(f'Assigning labels to the dataset')
        labels = pd.read_sql(f'SELECT date, close * shares_outstanding as marketcap, symbol '
                             f'FROM market.stock_prices '
                             f"WHERE symbol='{self.symbol}'", con=POSTGRES_URL)
        self.train_data = pd.merge(left=self.train_data, right=labels, on=['date', 'symbol'])
        logger.info(f'Training data is now of shape {self.train_data.shape} for {self.symbol} after assigning labels')

    def _train_test_split(self):
        logger.info(f'Whole dataset is of shape {self.data.shape} for {self.symbol}')
        logger.info(f'Splitting into train and test sets')
        self.train_data = self.data.loc[self.data['date'] < datetime.now()]
        self.test_data = self.data.loc[self.data['date'] >= datetime.now()]
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Testing data is of shape {self.test_data.shape}')


if __name__ == '__main__':
    # model = SingleModel('A')
    model = SectorModel(['A', 'AA'])
