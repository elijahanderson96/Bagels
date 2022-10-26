import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import logging
from config.configs import *
from config.common import QUERIES, SYMBOLS
from config.mappings import FINANCE_SECTOR, HEALTHCARE_SECTOR
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from data.transforms import FeaturePrepper
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelBase:
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_dates = None
        self.validation_dates = None
        self.test_dates = None
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.000003)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=5, mode='min', patience=25,
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

    def create_model(self):
        print(self.n_features)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_features, input_shape=(None, self.train_data.shape[1] - 2)),
            tf.keras.layers.Dense(int(self.n_features / 2), input_shape=(None, self.train_data.shape[1] - 2)),
            tf.keras.layers.Dense(int(self.n_features / 4), input_shape=(None, self.train_data.shape[1] - 2)),
            tf.keras.layers.Dense(int(self.n_features / 8), input_shape=(None, self.train_data.shape[1] - 2)),
            tf.keras.layers.Dense(1)])

    def normalize_train(self):
        self.train_dates = self.train_data.pop('date')
        self.columns = self.train_data.columns.to_list()
        self.train_data = self.scaler.fit_transform(self.train_data)

    def normalize_validation(self):
        self.validation_dates = self.validation_data.pop('date')
        self.validation_data = self.scaler.transform(self.validation_data)

    def normalize_test(self):
        self.test_dates = self.test_data.pop('date').to_list()
        self.test_data['marketcap'] = [1] * len(self.test_data)
        self.test_data = self.scaler.transform(self.test_data)

    def batch_train(self, batch_size=32):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data[:, :-1], self.train_data[:, -1]))
        self.train_data = self.train_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_validation(self, batch_size=1):
        self.validation_data = tf.data.Dataset.from_tensor_slices(
            (self.validation_data[:, :-1], self.validation_data[:, -1]))
        self.validation_data = self.validation_data.batch(batch_size, drop_remainder=True).batch(1, drop_remainder=True)

    def batch_test(self):
        self.test_data = tf.data.Dataset.from_tensor_slices(self.test_data[:, :-1])
        self.test_data = self.test_data.batch(1, drop_remainder=True).batch(1, drop_remainder=True)

    def train(self):
        self.create_model()
        self.model.summary()
        self.normalize_train()
        if self.validate: self.normalize_validation()
        self.batch_train()
        if self.validate: self.batch_validation()
        self.model.compile(optimizer=self.optimizer, loss='mse')
        with tf.device('/device:GPU:0'):
            self.history = self.model.fit(self.train_data, validation_data=self.validation_data, verbose=2, epochs=1000,
                                          callbacks=self.early_stopping)
        self.trained = True
        return self

    def predict(self):
        assert self.trained, 'Must train model before attempting prediction'
        self.normalize_test()
        self.batch_test()
        self.predictions = np.squeeze(self.model.predict(self.test_data))
        self.post_process()
        return self

    def save(self):
        self.model.save(f'./saved_models/model_{self.model_version}')

    def load(self, path_to_file):
        self.model = tf.keras.models.load_model(path_to_file)

    def post_process(self):
        data = pd.DataFrame(list(self.test_data.unbatch().unbatch().as_numpy_iterator()))
        data['marketcap'] = self.predictions
        self.predictions = pd.DataFrame(self.scaler.inverse_transform(data), columns=self.columns)
        self.market_cap_to_share_price()

    def market_cap_to_share_price(self):
        self.resolve_symbols()
        symbols = self.predictions['symbol'].unique()
        q = QUERIES['shares_outstanding'].replace('SYMBOLS',
                                                  f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        shares = pd.read_sql(q, con=POSTGRES_URL)
        tmp = []
        for symbol in symbols:
            n_shares = int(shares.loc[shares['symbol'] == symbol]['so'].unique().squeeze())
            tmp_df = self.predictions.loc[self.predictions['symbol'] == symbol].copy()
            tmp_df['close'] = tmp_df['marketcap'].apply(lambda mkcap: mkcap / n_shares)
            tmp.append(tmp_df)
        self.predictions = pd.concat(tmp, ignore_index=True)

    def resolve_symbols(self):
        tmp = []
        for col in self.sector:
            tmp_df = self.predictions.loc[self.predictions[col] == 1].copy()
            tmp_df['symbol'] = col
            tmp_df.drop(self.sector, axis=1, inplace=True)
            tmp.append(tmp_df)
        self.predictions = pd.concat(tmp, ignore_index=True).sort_values(by=['symbol'])
        self.predictions = self.predictions[['symbol', 'marketcap']]
        self.predictions['date'] = self.test_dates


class SectorModel(ModelBase):
    """Sector Model is used when building models for multiple symbols."""

    def __init__(self, sector, model_type):
        super().__init__()
        self.sector = sector
        self.model_type = model_type
        self.sector_string = ", ".join(self.sector)
        self.data = None
        self.validate = False
        self.interpolate_data = False
        self.interpolate_labels = False

    def train(self, validate=False, interpolate_data=False, interpolate_labels=False):
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
        self.gen_feature_matrix()
        return super().train()

    def predict(self):
        super().predict()

    def gen_feature_matrix(self):
        self.fetch_data()
        self.train_test_split()
        if self.validate: self.train_validation_split()
        self.transform_data()
        self.one_hot_encode()

    def fetch_data(self):
        logger.info(f'Fetching current data for {self.sector_string}')
        q = QUERIES['fundamental_valuations'].replace('SYMBOLS',
                                                      f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        self.data = pd.read_sql(q, con=POSTGRES_URL).rename(
            columns={'filingDate': 'date'})
        logger.info('Offsetting date by 91 days (13 weeks)')
        self.n_features = self.data.shape[1]
        self.data['date'] = pd.to_datetime(self.data['date']) + timedelta(days=91)
        self.data.replace(0, np.nan, inplace=True)
        self.data.dropna(how='all', axis=1, inplace=True)
        self.data.replace(np.nan, 0, inplace=True)

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

        return self.data

    def one_hot_encode(self):
        """This function will one hot encode the symbol column"""
        self.columns = [col for col in self.train_data.columns if col not in ('marketcap', 'symbol')]
        self.columns += self.sector + ['marketcap']
        self.train_data = pd.get_dummies(self.train_data, columns=['symbol'], prefix='', prefix_sep='')
        self.test_data = pd.get_dummies(self.test_data, columns=['symbol'], prefix_sep='', prefix='')
        if self.validate:
            self.validation_data = pd.get_dummies(self.validation_data, columns=['symbol'], prefix='', prefix_sep='')
        self.train_data = self.train_data[self.columns]
        self.validation_data = self.validation_data[self.columns]

    def assign_labels(self):
        logger.info(f'Assigning labels to the dataset')
        q = QUERIES['labels'].replace('SYMBOLS',
                                      f"{str(tuple(self.sector)) if len(self.sector) > 1 else str(tuple(self.sector)).replace(',)', ')')}")
        labels = pd.read_sql(q, con=POSTGRES_URL)
        self.train_data = self.train_data.merge(labels, on=['date', 'symbol'])
        logger.info(f'Training data is now of shape {self.train_data.shape} for {self.sector_string}')
        if self.validate:
            self.validation_data = self.validation_data.merge(labels, on=['date', 'symbol'])
            logger.info(f'Validation data is of shape {self.validation_data.shape}')

    def train_test_split(self):
        """Splits the dataset into training and test based on the current date. Since we are
        predicting prices for future dates, the train dataset is simply all data up to the current date."""
        logger.info(f'Splitting into train and test sets')
        self.train_data = self.data.loc[self.data['date'] < datetime.now()].sort_values(by=['symbol'])
        self.test_data = self.data.loc[self.data['date'] >= datetime.now()].sort_values(by=['symbol'])
        self.train_data = self.train_data.loc[self.train_data['symbol'].isin(self.test_data['symbol'])]
        idx = self.test_data.groupby('symbol')['date'].transform(max) == self.test_data['date']
        self.test_data = self.test_data[idx]
        self.sector = list(self.train_data['symbol'].unique())
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Testing data is of shape {self.test_data.shape}')

    def train_validation_split(self):
        logger.info('Splitting into train and validation sets')
        if len(self.train_data) > 2:
            self.train_data, self.validation_data = train_test_split(self.train_data, test_size=0.1)
        logger.info(f'Training data is of shape {self.train_data.shape}')
        logger.info(f'Validation data is of shape {self.validation_data.shape}')


if __name__ == '__main__':
    for symbol in SYMBOLS:
        model = SectorModel(sector=[symbol], model_type='fundamental_valuations')
        model.train(validate=True, interpolate_data=True, interpolate_labels=True)
        model.predict()
        model.save()
        val_loss = min(model.history.history['val_loss'])
        index = model.history.history['val_loss'].index(val_loss)
        training_loss = model.history.history['loss'][index]
        model.predictions['val_loss'] = val_loss
        model.predictions['loss'] = training_loss
        model.predictions.to_sql('predictions', con=POSTGRES_URL, schema='market', if_exists='append', index=False)

