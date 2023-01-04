import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config.common import QUERIES
from config.configs import *
from data.transforms import FeaturePrepper

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import psycopg2
from psycopg2.extensions import AsIs


class ModelBase:
    def __init__(self, stocks: list, validate: bool):
        self.stocks = stocks
        self.validate = validate

        self.train_data = None
        self.validation_data = None
        self.test_data = None

        self.train_dates = None
        self.validation_dates = None
        self.test_dates = None

        self.model = None
        self.model_size = None  # in gb

        self.trained = False
        self.predictions = None
        self.history = None
        self.n_features = None
        self.macro_queries = [
            'fetch_15Ymortgage_rates',
            'fetch_5Ymortgage_rates',
            'fetch_30Ymortgage_rates',
        ]

    def _fetch_macro_data(self):
        logger.info(f'Fetching macro economic data from database.')
        macro_dataframes = [pd.read_sql(QUERIES[q], con=POSTGRES_URL) for q in self.macro_queries]
        prepper = FeaturePrepper()
        macro_dataframes = prepper.transform_macro_data(macro_dataframes)
        return macro_dataframes

    def _create_features(self):
        """Features between regression and classification should be the same. The only
        difference are labels (market_cap (continuous) vs buy_or_sell (1 or 0)."""
        query = QUERIES['fundamental_valuations'].replace('SYMBOLS', f"{str(tuple(self.stocks))}")
        self.fundamental_valuation_data = pd.read_sql(query, con=POSTGRES_URL, parse_dates=['date', 'date_prev'])

        macro_data = self._fetch_macro_data()

        for df in macro_data:
            self.fundamental_valuation_data = self.fundamental_valuation_data.merge(df, on='date', how='left')

        self.fundamental_valuation_data.drop_duplicates(inplace=True, subset=['date', 'symbol'])

    def _train_test_split(self):
        """Splits the dataset into training and test based on the current date. Since we are
        predicting prices for future dates, the train dataset is simply all data up to the current date."""
        logger.info(f'Splitting into train and test sets')

        self.train_data = self.fundamental_valuation_data.loc[
            self.fundamental_valuation_data['date'] < datetime.now()].sort_values(by=['symbol'])

        self.test_data = self.fundamental_valuation_data.loc[
            self.fundamental_valuation_data['date'] >= datetime.now()].sort_values(by=['symbol'])

        #  subset the train data such that we only take symbols that are also in the test data
        self.train_data = self.train_data.loc[self.train_data['symbol'].isin(self.test_data['symbol'])]

        #  take the max date of each symbol as prediction/test data.
        mask = self.test_data.groupby('symbol')['date'].transform(max) == self.test_data['date']
        self.test_data = self.test_data[mask]
        self.stocks = list(self.test_data['symbol'].unique())

        if not self.validate:
            logger.info(f'Training data is of shape {self.train_data.shape}')
            logger.info(f'Testing data is of shape {self.test_data.shape}')

    def _train_validation_split(self):
        logger.info('Splitting into train and validation sets')

        #  we'll take 20 percent of data as validation data if we choose to validate
        if len(self.train_data) > 2:
            self.train_data, self.validation_data = train_test_split(self.train_data, test_size=0.2, shuffle=True)

        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Validation data is of shape {self.validation_data.shape}')
        logger.info(f'Testing data is of shape {self.test_data.shape}')

    def _create_datasets(self):
        self._create_model()
        self._create_features()
        self._train_test_split()

        if self.validate:
            self._train_validation_split()

        self._create_labels()

    def _create_labels(self):
        """Instantiated in child classes"""
        pass


class ClassificationModel(ModelBase):
    """Used to classify stocks to two possible taxonomies: Buy or Sell."""

    def __init__(self, stocks: list, validate=False):
        super().__init__(stocks, validate)

    def _create_model(self):
        self.model = tfdf.keras.GradientBoostedTreesModel(max_depth=6)

    def _create_labels(self):
        """Generate 1 or 0 based on whether the stock price is
        higher or lower 91 days after a quarterly report"""

        query = QUERIES['labels'].replace('SYMBOLS', f"{str(tuple(self.stocks))}")
        labels = pd.read_sql(query, con=POSTGRES_URL, parse_dates=['date'])

        self.train_data = self.train_data.merge(labels, on=['date', 'symbol'])

        labels.rename(columns={'marketcap': 'marketcap_prev',
                               'date': 'date_prev'}, inplace=True)

        self.train_data = self.train_data.merge(labels, on=['date_prev', 'symbol'])

        self.train_data['buy_sell'] = self.train_data['marketcap'] - self.train_data['marketcap_prev']
        self.train_data['buy_sell'] = self.train_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)

        self.train_data.drop(axis=1, inplace=True,
                             labels=['marketcap',
                                     'marketcap_prev',
                                     'entry_id',
                                     'date',
                                     'date_prev'])

        if self.validate:
            labels.rename(columns={'marketcap_prev': 'marketcap',
                                   'date_prev': 'date'}, inplace=True)

            self.validation_data = self.validation_data.merge(labels, on=['date', 'symbol'])

            labels.rename(columns={'marketcap': 'marketcap_prev',
                                   'date': 'date_prev'}, inplace=True)

            self.validation_data = self.validation_data.merge(labels, on=['date_prev', 'symbol'])
            self.validation_data['buy_sell'] = self.validation_data['marketcap'] - self.validation_data[
                'marketcap_prev']
            self.validation_data['buy_sell'] = self.validation_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)

            self.validation_data.drop(axis=1, inplace=True,
                                      labels=['marketcap',
                                              'marketcap_prev',
                                              'entry_id',
                                              'date',
                                              'date_prev'])

    def train(self):
        self._create_datasets()

        tf_train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_data, label="buy_sell")
        tf_valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.validation_data,
                                                                 label="buy_sell") if self.validate else None

        self.model.fit(tf_train_dataset, validation_data=tf_valid_dataset)
        self.model.summary()

        self.trained = True

        return self

    def predict(self):
        pass


class RegressionModel(ModelBase):
    """Used to try to predict a stock prices exact (continuous) value."""

    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               verbose=5,
                                                               mode='min',
                                                               patience=20,
                                                               restore_best_weights=True)

    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_features / 1.1, input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 1.5), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 1.75), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.01))])

    def gen_feature_matrix(self):
        self.fetch_data()
        self.train_test_split()
        if self.validate: self.train_validation_split()
        self.transform_data()
        if self.neural_net: self.one_hot_encode()
        self.n_features = self.train_data.shape[1]

    def transform_data(self):
        Prepper = FeaturePrepper()

        # if we want both data and labels interpolated, gotta assign labels first, then transform
        if self.interpolate_data and self.interpolate_labels:
            self.assign_labels()
            self.train_data = Prepper.transform(self.train_data, interpolate=True).sample(frac=1)
            logger.info(f'Shape of training data after interpolate is {self.train_data.shape}')
        # otherwise if we only want data, transform, then assign labels.
        if self.interpolate_data and not self.interpolate_labels:
            self.train_data = Prepper.transform(self.train_data, interpolate=True)
            self.assign_labels()

        if not self.interpolate_data and not self.interpolate_labels:
            self.assign_labels()

        return self.fundamental_valuation_data

    def one_hot_encode(self):
        """This function will one hot encode the symbol column"""
        self.columns = [col for col in self.train_data.columns if col not in ('marketcap', 'symbol')]
        self.columns += self.sector + ['marketcap']
        self.train_data = pd.get_dummies(self.train_data, columns=['symbol'], prefix='', prefix_sep='')
        self.test_data = pd.get_dummies(self.test_data, columns=['symbol'], prefix_sep='', prefix='')
        if self.validate:
            self.validation_data = pd.get_dummies(self.validation_data, columns=['symbol'], prefix='', prefix_sep='')
        self.train_data = self.train_data[self.columns]

        # validation data is randomly sampled and may not have all symbols. This loop ensures
        # that all symbols are one hot encoded (which is required because data must be of same shape
        for col in self.columns:
            if col not in self.validation_data.columns:
                self.validation_data[col] = 0
        self.validation_data = self.validation_data[self.columns]

    def _create_labels(self):
        logger.info(f'Assigning labels to the dataset')
        q = QUERIES['labels'].replace('SYMBOLS',
                                      f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        labels = pd.read_sql(q, con=POSTGRES_URL)
        self.train_data = self.train_data.merge(labels, on=['date', 'symbol'])

        logger.info(f'Training data is now of shape {self.train_data.shape} for {self.sector_string}')

        self.validation_data = self.validation_data.merge(labels, on=['date', 'symbol'])
        logger.info(f'Validation data is of shape {self.validation_data.shape}')
        self.validation_data = self.validation_data[
            [col for col in self.validation_data.columns if col != 'marketcap']]

        self.train_data = self.train_data[
            [col for col in self.train_data.columns if col != 'marketcap']]

    def normalize_training_data(self):
        self.train_dates = self.train_data.pop('date').to_list()
        self.metadata = {col: df.pop('entry_id').to_list() if isinstance(df, pd.DataFrame)
                                                              and not df.empty else None
                         for col, df in {'train_ids': self.train_data,
                                         'validation_ids': self.validation_data,
                                         'test_ids': self.test_data}.items()}
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def normalize_validation_data(self):
        self.validation_dates = self.validation_data.pop('date').to_list()
        self.validation_data = self.scaler.transform(self.validation_data)

    def normalize_test_data(self):
        self.test_dates = self.test_data.pop('date').to_list()
        self.test_data['marketcap'] = [1] * len(self.test_data)
        self.test_data = self.scaler.transform(self.test_data)

    def batch_training_data(self, batch_size=4):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1], self.train_data[:, -1]))
        self.train_data = self.train_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_validation_data(self, batch_size=1):
        self.validation_data = tf.data.Dataset.from_tensor_slices(
            (self.validation_data[:, :-1], self.validation_data[:, -1]))
        self.validation_data = self.validation_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test_data(self):
        self.test_data = tf.data.Dataset.from_tensor_slices(self.test_data[:, :-1])
        self.test_data = self.test_data.batch(1, drop_remainder=True).batch(1, drop_remainder=True)

    def train(self):
        self.create_model()
        self.normalize_training_data()
        self.batch_training_data()

        if self.validate:
            self.normalize_validation_data()
            self.batch_validation_data()

        self.model.compile(optimizer=self.optimizer, loss='mse')

        with tf.device('/device:GPU:0'):
            self.history = self.model.fit(self.train_data,
                                          validation_data=self.validation_data,
                                          verbose=2,
                                          epochs=10000,
                                          callbacks=self.early_stopping)

        self.trained = True
        self.metadata['date_created'] = datetime.now()

        if self.validate:
            val_loss = min(self.history.history['val_loss'])
            index = self.history.history['val_loss'].index(val_loss)

        training_loss = self.history.history['loss'][index]

        self.metadata['val_loss'] = val_loss
        self.metadata['loss'] = training_loss

    def post_process(self):
        prediction_data = pd.DataFrame(list(self.test_data.unbatch().unbatch().as_numpy_iterator()),
                                       columns=self.columns[:-1])
        training_data = pd.DataFrame([np.append(arr=i[0], values=i[1])
                                      for i in list(self.train_data.unbatch().unbatch().as_numpy_iterator())],
                                     columns=self.columns)
        validation_data = pd.DataFrame([np.append(arr=i[0], values=i[1]) for i in
                                        list(self.validation_data.unbatch().unbatch().as_numpy_iterator())],
                                       columns=self.columns)

        prediction_data['marketcap'] = self.predictions.flatten()
        training_data['marketcap'] = self.train_predictions.flatten()
        validation_data['marketcap'] = self.valid_predictions.flatten()

        self.predictions = pd.DataFrame(self.scaler.inverse_transform(prediction_data), columns=self.columns)
        self.predictions['date'] = self.test_dates[0:len(self.predictions)]
        self.predictions['entry_id'] = self.metadata['test_ids'][0:len(self.predictions)]
        self.predictions['scores'] = 'test_scores'
        self.metadata['test_ids'] = self.metadata['test_ids'][0:len(self.predictions)]

        self.train_predictions = pd.DataFrame(self.scaler.inverse_transform(training_data), columns=self.columns)
        self.train_predictions['date'] = self.train_dates[0:len(self.train_predictions)]
        self.train_predictions['entry_id'] = self.metadata['train_ids'][0:len(self.train_predictions)]
        self.train_predictions['scores'] = 'train_scores'

        self.metadata['train_ids'] = self.metadata['train_ids'][0:len(self.train_predictions)]

        self.validation_predictions = pd.DataFrame(self.scaler.inverse_transform(validation_data), columns=self.columns)
        self.validation_predictions['date'] = self.validation_dates[0:len(self.validation_predictions)]
        self.validation_predictions['entry_id'] = self.metadata['validation_ids'][0:len(self.validation_predictions)]
        self.validation_predictions['scores'] = 'validation_scores'

        self.metadata['validation_ids'] = self.metadata['validation_ids'][0:len(self.validation_predictions)]

        self.all_scores = pd.concat([self.predictions, self.train_predictions, self.validation_predictions],
                                    ignore_index=True)
        self.market_cap_to_share_price()
        self.save_model_scores()

    def market_cap_to_share_price(self):
        self.resolve_symbols()
        symbols = self.all_scores['symbol'].unique()
        q = QUERIES['shares_outstanding'].replace('SYMBOLS',
                                                  f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        shares = pd.read_sql(q, con=POSTGRES_URL)
        tmp = []

        for symbol in symbols:
            n_shares = shares.loc[shares['symbol'] == symbol]['so'].unique().tolist()[0]
            tmp_df = self.all_scores.loc[self.all_scores['symbol'] == symbol].copy()
            tmp_df['close'] = tmp_df['marketcap'].apply(lambda mkcap: mkcap / n_shares)
            tmp_df['n_shares'] = n_shares
            tmp.append(tmp_df)
        self.all_scores = pd.concat(tmp, ignore_index=True)

    def resolve_symbols(self):
        tmp = []
        for col in self.sector:
            tmp_df = self.all_scores.loc[self.all_scores[col] == 1].copy()
            tmp_df['symbol'] = col
            tmp_df.drop(self.sector, axis=1, inplace=True)
            tmp.append(tmp_df)
        self.all_scores = pd.concat(tmp, ignore_index=True).sort_values(by=['symbol'])

    def save_model_scores(self):
        """Updates metadata to include training, validation, and prediction results by
        calling model.predict on the training, validation and prediction data. We will store
        this in db and look back on the results to gauge accuracy"""
        # need model predictions and actual series values
        tmp = {}
        scores = ('train_scores', 'validation_scores', 'test_scores')
        self.metadata['model_type'] = self.model_type
        for score in scores:
            for symbol in self.all_scores['symbol'].unique():
                score_df = \
                    self.all_scores.loc[(self.all_scores['scores'] == score) & (self.all_scores['symbol'] == symbol)][
                        ['date', 'close']]
                tmp[symbol] = {str(date): prediction for date, prediction in
                               dict(zip(score_df['date'], score_df['close'])).items()}
            tmp = json.dumps(tmp)
            self.metadata[score] = tmp
            tmp = {}
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        columns = self.metadata.keys()
        values = [self.metadata[column] for column in columns]
        insert_statement = 'insert into market.models (%s) values %s'
        cursor.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))
        conn.commit()
        cursor.close()
        conn.close()


class PredictionPipeline:
    def __init__(self, symbols, model_type='', validate=False):
        assert model_type.lower() in ('classify', 'regression')
        self.symbols = symbols
        self.validate = validate
        self.model = ClassificationModel(symbols, validate) if model_type.lower() == 'classify' else \
            RegressionModel(symbols, validate)

    def train(self):
        return self.model.train()

    def predict(self):
        return self.model.predict()


self = ClassificationModel(['BAC', 'C'])
