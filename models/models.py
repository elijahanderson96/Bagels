import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from config.configs import *
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

load_dotenv()

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FeaturePrepper:
    """Takes data from any endpoint and transform it. We impute row data such that
    any zeros are replaced by the average value of the row, and interpolate the data such
    that there's data for every day. It is a linear interpolation."""

    def __init__(self, data):
        self.original_data = data
        self.chunked_data = None
        self.labeled_data = None
        self.symbols = None
        self.report_dates = None
        self.preprocess()

    @staticmethod
    def interpolate(df_chunk):
        assert df_chunk['symbol'].iloc[0] == df_chunk['symbol'].iloc[1], 'symbols do not match'
        df_chunk['reportdate'] = df_chunk['reportdate'].astype('datetime64')
        n_days = abs((df_chunk['reportdate'].iloc[0] - df_chunk['reportdate'].iloc[1]).days)
        dates = pd.date_range(df_chunk['reportdate'].iloc[0], df_chunk['reportdate'].iloc[1], freq='d')
        temp_values = {}
        for col in df_chunk.columns:
            if col not in ('reportdate', 'symbol'):
                tmp = []
                for j in range(n_days + 1):
                    number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                    tmp.append(number)
                temp_values.update({col: tmp})
        temp_values.update({'date': dates, 'symbol': df_chunk['symbol'].iloc[0]})
        return pd.DataFrame(temp_values)

    def preprocess(self):
        """Sorts the values by symbols (if multiple) and report dates. Pops the report dates (after adding 91 days)
        and symbols into a class attribute, and returns all numeric data from the original data"""
        self.original_data.sort_values(by=['symbol', 'reportdate'], inplace=True)
        self.original_data.drop_duplicates(inplace=True, subset=['symbol', 'reportdate'])  # add symbol
        self.original_data['reportdate'] = pd.to_datetime(self.original_data['reportdate']) + timedelta(days=91)
        report_dates = self.original_data['reportdate'].to_list()
        symbols = self.original_data['symbol'].to_list()
        self.original_data = self.original_data._get_numeric_data()
        self.original_data['reportdate'] = report_dates
        self.original_data['symbol'] = symbols
        # keep only the columns in df that do not contain string
        self.chunked_data = [self.original_data.iloc[i:i + 2] for i in range(len(self.original_data) - 1)
                             if self.original_data['symbol'].iloc[i] == self.original_data['symbol'].iloc[i + 1]]

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df

    def transform(self):
        logger.info(f'Provisioning {cpu_count() - 1} cpus for transformation')
        with Pool(cpu_count() - 1) as p:
            dfs = p.map(self.interpolate, self.chunked_data)
            df = pd.concat(dfs, ignore_index=True)
        df = self._impute_row_data(df)
        logger.info(f'Interpolation for {df["symbol"].unique()} complete')
        return df


class ModelBase:
    def __init__(self):
        super().__init__()
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
        self.symbols = None

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
        columns = [col for col in self.train_data.columns if col not in ('marketcap', 'symbol')]
        self.symbols = [col for col in self.sector if col in self.train_data['symbol'].to_list()]
        total_cols = columns + self.symbols
        self.train_data = pd.get_dummies(self.train_data, columns=['symbol'], prefix='', prefix_sep='')
        self.test_data = pd.get_dummies(self.test_data, columns=['symbol'], prefix_sep='', prefix='')
        total_cols.extend(['marketcap'])
        self.train_data = self.train_data[total_cols]
        print(self.train_data.sample(15).iloc[: -8:])
        input('break')

    def train(self):
        if self.sector: self.one_hot_encode()  # dont need to one hot encode single models
        self.normalize_train()
        self.create_model()
        self.batch_train()
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.fit(self.train_data, verbose=2, epochs=3, callbacks=self.early_stopping)
        self.trained = True
        return self

    def predict(self):
        assert self.trained, 'Must train model before attempting prediction'
        self.normalize_test()
        self.batch_test()
        self.predictions = self.model.predict(self.test_data)
        self.post_process()
        return self

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
        if self.symbol:
            self.predictions = self.predictions[['symbol', 'date', 'marketcap']]
        else:
            self._resolve_symbols()  # have to cast back after one hot encoding

        symbols = self.predictions['symbol'].unique()
        shares = pd.read_sql(f"SELECT symbol, shares_outstanding as so FROM market.stock_prices "
                             f" WHERE symbol in {tuple(symbols)}",
                             con=POSTGRES_URL)
        tmp = []
        for symbol in symbols:
            n_shares = int(shares.loc[shares['symbol'] == symbol]['so'].unique().squeeze())
            tmp_df = self.predictions.loc[self.predictions['symbol'] == symbol]
            tmp_df['close'] = tmp_df['marketcap'].apply(lambda mkcap: mkcap / n_shares)
            tmp.append(tmp_df)
        self.predictions = pd.concat(tmp, ignore_index=True)

    def _resolve_symbols(self):
        tmp = []
        for col in self.symbols:  # for each symbol, go through and find where its equal to one.
            tmp_df = self.predictions.loc[self.predictions[col] == 1]
            tmp_df['symbol'] = col
            tmp_df.drop(self.symbols, axis=1, inplace=True)
            tmp.append(tmp_df)
        self.predictions = pd.concat(tmp, ignore_index=True)[['symbol', 'date', 'marketcap']]


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
        self.data = pd.read_sql(f'SELECT * FROM market.fundamentals '  # TODO MAKE DYNAMIC
                                f'ORDER BY symbol, date ASC; ', con=POSTGRES_URL)
        print(self.data)
        # self.data.drop(inplace=True,columns=['date'])
        # self.data.rename(columns={'reportdate':'date'},inplace=True)
        # self.data['date'] = self.data['date'].astype('datetime64')
        self.data = FeaturePrepper(self.data).transform()
        print(self.data)
        self._train_test_split()
        self._assign_labels()

    def _assign_labels(self):
        logger.info(f'Assigning labels to the dataset')
        labels = pd.read_sql(f'SELECT date, close * shares_outstanding as marketcap, symbol '
                             f'FROM market.stock_prices', con=POSTGRES_URL)
        self.train_data = pd.merge(left=self.train_data, right=labels, on=['date', 'symbol'])
        print(self.train_data.sample(15).iloc[:, -8:])
        input('break')
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
        self.data = pd.read_sql(f'SELECT * FROM market.fundamentals '
                                f'ORDER BY date ASC; ', con=POSTGRES_URL)
        self.data['date'] = self.data['date'].astype('datetime64[ns]')
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
    print('hello')
    from config.common import SYMBOLS
    model = SectorModel(sector=SYMBOLS)
    print('instantiated')
    model.train()
    print('fitted')
    model.predict()
    print(model.predictions)
    model.predictions.to_sql('predictions', con=POSTGRES_URL, schema='market', if_exists='replace', index=False,)

