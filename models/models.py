import json
import logging
from datetime import datetime, timedelta

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
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_dates = None
        self.validation_dates = None
        self.test_dates = None
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               verbose=5,
                                                               mode='min',
                                                               patience=20,
                                                               restore_best_weights=True)
        self.model_size = None  # in gb
        self.columns = None
        self.scaler = MinMaxScaler()
        self.predictions = None
        self.trained = False
        self.validate = False
        self.symbols = None
        self.model_version = str(datetime.now()).replace(' ', '_')
        self.model_tester_data = []
        self.history = None
        self.n_features = None
        self.macro_queries = [
            # 'fetch_real_gdp', 'fetch_fed_funds', 'fetch_comm_paper_outstanding',
            ##'fetch_unemployment_claims', 'fetch_cpi', 'fetch_vehicle_sales', 'fetch_unemployment_rate',
            # 'fetch_industrial_production', 'fetch_housing_starts', 'fetch_num_total_employees',
            # 'fetch_recession_probability',
            'fetch_15Ymortgage_rates', 'fetch_5Ymortgage_rates',
            'fetch_30Ymortgage_rates',
        ]

    def create_deep_neural_net_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_features / 1.1, input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features / 1.5), input_shape=(None, self.train_data.shape[1] - 3),
                                  activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(int(self.n_features/ 1.75), input_shape=(None, self.train_data.shape[1] - 3),
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

    def normalize_train(self):
        self.train_dates = self.train_data.pop('date').to_list()
        self.metadata = {col: df.pop('entry_id').to_list() if isinstance(df, pd.DataFrame)
                                                              and not df.empty else None
                         for col, df in {'train_ids': self.train_data,
                                         'validation_ids': self.validation_data,
                                         'test_ids': self.test_data}.items()}
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def normalize_validation(self):
        self.validation_dates = self.validation_data.pop('date').to_list()
        self.validation_data = self.scaler.transform(self.validation_data)

    def normalize_test(self):
        self.test_dates = self.test_data.pop('date').to_list()
        self.test_data['marketcap'] = [1] * len(self.test_data)
        self.test_data = self.scaler.transform(self.test_data)

    def batch_train(self, batch_size=4):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1], self.train_data[:, -1]))
        self.train_data = self.train_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_validation(self, batch_size=1):
        self.validation_data = tf.data.Dataset.from_tensor_slices(
            (self.validation_data[:, :-1], self.validation_data[:, -1]))
        self.validation_data = self.validation_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test(self):
        self.test_data = tf.data.Dataset.from_tensor_slices(self.test_data[:, :-1])
        self.test_data = self.test_data.batch(1, drop_remainder=True).batch(1, drop_remainder=True)

    def train(self, classify=False, boosted_trees=True, neural_net=False):
        if neural_net:
            self.create_deep_neural_net_model()
            self.normalize_train()
            if self.validate: self.normalize_validation()
            self.batch_train()
            if self.validate: self.batch_validation()
            self.model.compile(optimizer=self.optimizer, loss='mse')
        if boosted_trees:
            print(self.train_data)
            tf_train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_data, label="buy_sell")
            if self.validate:
                tf_valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.validation_data, label="buy_sell")
            else:
                tf_valid_dataset = None
            self.model = tfdf.keras.GradientBoostedTreesModel()
            self.model.fit(tf_train_dataset, validation_data=tf_valid_dataset)

        self.model.summary()
        #with tf.device('/device:GPU:0'):
        #    self.history = self.model.fit(self.train_data, validation_data=self.validation_data, verbose=2,
        #                                  epochs=10000,
        #                                  callbacks=self.early_stopping)
        self.trained = True
        #self.metadata['date_created'] = datetime.now()
        #if self.validate:
        ##    val_loss = min(self.history.history['val_loss'])
        #    index = self.history.history['val_loss'].index(val_loss)
        #training_loss = self.history.history['loss'][index]
        #self.metadata['val_loss'] = val_loss
        #self.metadata['loss'] = training_loss
        return self

    def predict(self):
        assert self.trained, 'Must train model before attempting prediction'
        self.normalize_test()
        self.batch_test()
        self.predictions = np.squeeze(self.model.predict(self.test_data))
        self.train_predictions = np.squeeze(self.model.predict(self.train_data))
        self.valid_predictions = np.squeeze(self.model.predict(self.validation_data))
        self.post_process()

        return self

    def save(self):
        self.model.save(f'./saved_models/model_{self.model_version}')

    def load(self, path_to_file):
        self.model = tf.keras.models.load_model(path_to_file)

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


class SectorModel(ModelBase):
    """Sector Model is used when building models for multiple symbols."""

    def __init__(self, sector, model_type):
        super().__init__()
        self.sector = sector
        self.model_type = model_type
        self.sector_string = ", ".join(self.sector)
        self.fundamental_valuation_data = None
        self.validate = False
        self.interpolate_data = False
        self.interpolate_labels = False

    def train(self, validate=False, interpolate_data=False, interpolate_labels=False, neural_net=False):
        """
        Train the model.
        Args:
            validate: Whether to hold out a single data point (the latest quarterly report) to validate against
            interpolate_data: Whether we want to linearly interpolate data between quarterly report dates
            interpolate_labels: Whether we want to linearly interpolate labels between quarterly report dates,
            or use the actual closing prices.

        Returns:
            self
        """
        self.validate = validate
        self.interpolate_data = interpolate_data
        self.interpolate_labels = interpolate_labels
        self.neural_net = neural_net
        self.gen_feature_matrix()
        return super().train()

    def predict(self):
        pass
        #super().predict()

    def gen_feature_matrix(self):
        self.fetch_data()
        self.train_test_split()
        if self.validate: self.train_validation_split()
        self.transform_data()
        if self.neural_net: self.one_hot_encode()
        self.n_features = self.train_data.shape[1]

    def fetch_data(self):
        logger.info(f'Fetching current data for {self.sector_string}')
        q = QUERIES['fundamental_valuations'].replace('SYMBOLS',
                                                      f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        self.fundamental_valuation_data = pd.read_sql(q, con=POSTGRES_URL)
        logger.info('Offsetting date by 91 days (13 weeks)')
        self.fundamental_valuation_data['date_prev'] = self.fundamental_valuation_data['date']
        self.fundamental_valuation_data['date'] = pd.to_datetime(self.fundamental_valuation_data['date']) + timedelta(
            days=91)

        self.fundamental_valuation_data.replace(0, np.nan, inplace=True)
        self.fundamental_valuation_data.dropna(how='all', axis=1, inplace=True)
        self.fundamental_valuation_data.replace(np.nan, 0, inplace=True)

        macro_data = self.fetch_macro_data()
        for df in macro_data:
            self.fundamental_valuation_data = self.fundamental_valuation_data.merge(df, on='date', how='left')
        self.fundamental_valuation_data.drop_duplicates(inplace=True, subset=['date', 'symbol'])
        self.fundamental_valuation_data.replace(np.nan, 0, inplace=True)

    def fetch_macro_data(self):
        logger.info(f'Fetching macro economic data from database.')
        macro_dataframes = [pd.read_sql(QUERIES[q], con=POSTGRES_URL) for q in self.macro_queries]
        prepper = FeaturePrepper()
        macro_dataframes = prepper.transform_macro_data(macro_dataframes)
        return macro_dataframes

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

    def assign_labels(self):
        logger.info(f'Assigning labels to the dataset')
        q = QUERIES['labels'].replace('SYMBOLS',
                                      f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        labels = pd.read_sql(q, con=POSTGRES_URL)
        self.train_data = self.train_data.merge(labels, on=['date', 'symbol'])
        labels.rename(columns={'marketcap': 'marketcap_prev', 'date':'date_prev'}, inplace=True)
        self.train_data['date_prev'] = self.train_data['date_prev'].astype('datetime64[ns]')
        self.train_data = self.train_data.merge(labels, on=['date_prev', 'symbol'])
        self.train_data['buy_sell'] = self.train_data['marketcap'] - self.train_data['marketcap_prev']
        self.train_data['buy_sell'] = self.train_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)
        self.train_data.drop(axis=1, inplace=True, labels=['marketcap','marketcap_prev','entry_id','date','date_prev'])
        logger.info(f'Training data is now of shape {self.train_data.shape} for {self.sector_string}')
        if self.validate:
            labels.rename(columns={'marketcap_prev': 'marketcap', 'date_prev': 'date'}, inplace=True)
            self.validation_data = self.validation_data.merge(labels, on=['date', 'symbol'])
            labels.rename(columns={'marketcap': 'marketcap_prev', 'date': 'date_prev'}, inplace=True)
            self.validation_data['date_prev'] = self.validation_data['date_prev'].astype('datetime64[ns]')
            self.validation_data = self.validation_data.merge(labels, on=['date_prev', 'symbol'])
            print(self.validation_data)
            self.validation_data['buy_sell'] = self.validation_data['marketcap'] - self.validation_data['marketcap_prev']
            self.validation_data['buy_sell'] = self.validation_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)
            self.validation_data.drop(axis=1, inplace=True,labels=['marketcap', 'marketcap_prev', 'entry_id', 'date', 'date_prev'])

            logger.info(f'Validation data is of shape {self.validation_data.shape}')
        self.train_data = self.train_data[
            [col for col in self.train_data.columns if col != 'marketcap']]
        if self.validate:
            self.validation_data = self.validation_data[
            [col for col in self.validation_data.columns if col != 'marketcap']]

    def train_test_split(self):
        """Splits the dataset into training and test based on the current date. Since we are
        predicting prices for future dates, the train dataset is simply all data up to the current date."""
        logger.info(f'Splitting into train and test sets')
        self.train_data = self.fundamental_valuation_data.loc[
            self.fundamental_valuation_data['date'] < datetime.now()].sort_values(by=['symbol'])
        self.test_data = self.fundamental_valuation_data.loc[
            self.fundamental_valuation_data['date'] >= datetime.now()].sort_values(by=['symbol'])
        self.train_data = self.train_data.loc[self.train_data['symbol'].isin(self.test_data['symbol'])]
        idx = self.test_data.groupby('symbol')['date'].transform(max) == self.test_data['date']
        self.test_data = self.test_data[idx]
        self.sector = list(self.test_data['symbol'].unique())
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Testing data is of shape {self.test_data.shape}')

    def train_validation_split(self):
        logger.info('Splitting into train and validation sets')
        if len(self.train_data) > 2:
            self.train_data, self.validation_data = train_test_split(self.train_data, test_size=0.1, shuffle=True)
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Validation data is of shape {self.validation_data.shape}')
