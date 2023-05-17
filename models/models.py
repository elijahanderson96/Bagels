import logging
from datetime import datetime, timedelta

import pandas as pd
import tensorflow_decision_forests as tfdf

from config.configs import POSTGRES_URL
from data.transforms import FeaturePrepper

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ClassificationModel:
    """Used to classify stocks to two possible taxonomies: Buy or Sell."""

    def __init__(self, stocks: list, validate=False, features=None):
        self.stocks = stocks
        self.validate = validate
        self.features = features

    def _create_model(self):
        tuner = tfdf.tuner.RandomSearch(num_trials=100)
        tuner.choice("min_examples", [2, 5, 7, 10, 14])
        tuner.choice("categorical_algorithm", ["CART", "RANDOM"])
        local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
        local_search_space.choice("max_depth", [3, 4, 5, 6, 8, 9])
        global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
        global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256, 512])
        tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
        tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])
        tuner.choice("split_axis", ["AXIS_ALIGNED"])
        oblique_space = tuner.choice("split_axis", ["SPARSE_OBLIQUE"], merge=True)
        oblique_space.choice("sparse_oblique_normalization", ["NONE", "STANDARD_DEVIATION", "MIN_MAX"])
        oblique_space.choice("sparse_oblique_weights", ["BINARY", "CONTINUOUS"])
        oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])
        self.model = tfdf.keras.RandomForestModel()

    def _create_features(self):
        """Pass a dict of datasets that we will join together by date
        to form a feature matrix.

        Args:
            datasets: dict of datasets with key being query name, value being any
            query parameters. Example {'fundamental_valudations': {'symbol'

        """
        feature_generator = FeaturePrepper(self.features, {'SYMBOLS': str(tuple(self.stocks))})
        self.data = feature_generator.create_feature_matrix()
        self.data.sort_values(by=['symbol', 'date'], inplace=True)
        labels = self._create_labels()
        self.data['date'] = (self.data['date'] - timedelta(days=7))
        self.data = self.data.merge(labels, on=['symbol', 'date'])
        #columns = [col for col in self.data.columns if col not in ('symbol', 'date')]

        diff_df = self.data.groupby('symbol')['close'].diff()
        diff_df[diff_df['close'] < 0] = 0
        diff_df[diff_df['close'] > 0] = 1
        diff_df['close'] = diff_df['close'].shift(-1)
        diff_df['symbol'] = self.data['symbol']
        self.train_data = pd.concat([self.data, diff_df]).dropna()
        self.n_features = self.data.shape[1]

    def _create_labels(self):
        return pd.read_sql('SELECT close, symbol,"priceDate" as date FROM bagels.labels WHERE symbol IN %(symbols)s',
                           params={'symbols': tuple(self.stocks)}, con=POSTGRES_URL)

    def _train_test_split(self):
        """Splits the dataset into training and test based on the current date. Since we are
        predicting prices for future dates, the train dataset is simply all data up to the current date."""
        logger.info(f'Splitting into train and test sets')
        self.train_data = self.data.loc[self.data['date'] < datetime.now()].sort_values(by=['symbol'])
        self.test_data = self.data.loc[self.data['date'] >= datetime.now()].sort_values(by=['symbol'])
        logger.info(f'Training data is of shape {self.train_data.shape}')

        if self.test_data.empty:
            logger.info('No test data.')
            return self.test_data

        # take the max date of each symbol as prediction/test data.
        mask = self.test_data.groupby('symbol')['date'].transform(max) == self.test_data['date']
        logger.info(f'Testing data is of shape {self.test_data.shape}')
        self.test_data = self.test_data[mask]
        self.stocks = list(self.test_data['symbol'].unique())

    def _create_datasets(self):
        self._create_model()
        self._create_features()
        #self._train_test_split()

    def train(self):
        self._create_datasets()
        tf_train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_data, label="close")
        self.model.fit(tf_train_dataset)
        #tuning_logs = self.model.make_inspector().tuning_logs()
        #tuning_logs.head()
        self.model.summary()
        return self

    def predict(self):
        to_drop = ['date', 'date_prev']
        test_data = self.test_data.drop(axis=1, labels=to_drop)
        tf_test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data)
        self.test_data['predictions'] = self.model.predict(tf_test_dataset)
        return


class PredictionPipeline:
    def __init__(self, symbols, model_type='', validate=False, features=None):
        assert model_type.lower() in ('classify', 'regression')
        self.symbols = symbols
        self.validate = validate
        self.model = ClassificationModel(symbols, validate, features)

    def train(self):
        return self.model.train()

    def predict(self):
        return self.model.predict()

