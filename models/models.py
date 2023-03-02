import logging
from datetime import datetime

import pandas as pd
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

from config.configs import POSTGRES_URL
from config.common import QUERIES
from data.transforms import FeaturePrepper

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelBase:
    def __init__(self, stocks: list, validate: bool, features: list):
        self.stocks = stocks
        self.validate = validate
        self.features = features

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

    def _create_features(self):
        """Pass a dict of datasets that we will join together by date
        to form a feature matrix.

        Args:
            datasets: dict of datasets with key being query name, value being any
            query parameters. Example {'fundamental_valudations': {'symbol'

        """
        feature_generator = FeaturePrepper(self.features, {'SYMBOLS': str(tuple(self.stocks))})
        self.data = feature_generator.create_feature_matrix()
        self.n_features = self.data.shape[1]

    def _train_test_split(self):
        """Splits the dataset into training and test based on the current date. Since we are
        predicting prices for future dates, the train dataset is simply all data up to the current date."""
        logger.info(f'Splitting into train and test sets')

        self.train_data = self.data.loc[
            self.data['date'] < datetime.now()].sort_values(by=['symbol'])

        self.test_data = self.data.loc[
            self.data['date'] >= datetime.now()].sort_values(by=['symbol'])

        #  take the max date of each symbol as prediction/test data.
        mask = self.test_data.groupby('symbol')['date'].transform(max) == self.test_data['date']
        logger.info(f'Testing data is of shape {self.test_data.shape}')
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

    def _create_model(self):
        # instantiated in child class
        pass


class ClassificationModel(ModelBase):
    """Used to classify stocks to two possible taxonomies: Buy or Sell."""

    def __init__(self, stocks: list, validate=False, features=None):
        super().__init__(stocks, validate, features)

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
        oblique_space.choice("sparse_oblique_normalization",
                             ["NONE", "STANDARD_DEVIATION", "MIN_MAX"])
        oblique_space.choice("sparse_oblique_weights", ["BINARY", "CONTINUOUS"])
        oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])

        self.model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)

    def _create_labels(self):
        """Generate 1 or 0 based on whether the stock price is
        higher or lower 91 days after a quarterly report"""

        query = QUERIES['labels'].replace('SYMBOLS', f"{str(tuple(self.stocks))}")
        labels = pd.read_sql(query, con=POSTGRES_URL, parse_dates=['date'])
        self.train_data['date_prev'] = self.train_data['date_prev'].astype('datetime64[ns]')

        self.train_data = self.train_data.merge(labels, on=['date', 'symbol'])
        labels.rename(columns={'marketcap': 'marketcap_prev', 'date': 'date_prev'}, inplace=True)
        self.train_data = self.train_data.merge(labels, on=['date_prev', 'symbol'])

        self.train_data['buy_sell'] = self.train_data['marketcap'] - self.train_data['marketcap_prev']
        self.train_data['buy_sell'] = self.train_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)

        to_drop = ['marketcap', 'marketcap_prev', 'date', 'date_prev']
        self.train_data.drop(axis=1, inplace=True, labels=to_drop)

        if self.validate:
            self.validation_data['date_prev'] = self.validation_data['date_prev'].astype('datetime64[ns]')
            labels.rename(columns={'marketcap_prev': 'marketcap', 'date_prev': 'date'}, inplace=True)
            self.validation_data = self.validation_data.merge(labels, on=['date', 'symbol'])
            labels.rename(columns={'marketcap': 'marketcap_prev', 'date': 'date_prev'}, inplace=True)

            self.validation_data = self.validation_data.merge(labels, on=['date_prev', 'symbol'])
            self.validation_data['buy_sell'] = self.validation_data['marketcap'] - self.validation_data[
                'marketcap_prev']
            self.validation_data['buy_sell'] = self.validation_data['buy_sell'].apply(lambda row: 1 if row > 0 else 0)

            self.validation_data.drop(axis=1, inplace=True, labels=to_drop)

    def train(self):
        self._create_datasets()

        tf_train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_data, label="buy_sell")
        tf_valid_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(self.validation_data,
                                                                 label="buy_sell") if self.validate else None
        self.model.fit(tf_train_dataset, validation_data=tf_valid_dataset)
        tuning_logs = self.model.make_inspector().tuning_logs()
        tuning_logs.head()
        self.model.summary()

        self.trained = True

        return self

    @staticmethod
    def recommender(prediction_results):
        #  price to book value between 0 and 4, and prediction value of .5 or greater is what we look to invest in.
        price_to_bookvalue_and_equity = pd.read_sql(
            'SELECT symbol, '
            '"filingDate" as date, '
            '"pToBv", '
            '"pToE" FROM market.fundamental_valuations '
            f'WHERE symbol IN {str(tuple(prediction_results["symbol"]))} '
            f'ORDER BY "filingDate" DESC '
            f'LIMIT {len(prediction_results)}', con=POSTGRES_URL)

        prediction_results = prediction_results.merge(price_to_bookvalue_and_equity, on='symbol').sort_values(
            by=['predictions', 'pToBv']).drop_duplicates()

        closing_prices = pd.read_sql(f'SELECT symbol, date, close '
                                     f'FROM market.stock_prices '
                                     f'WHERE symbol in {str(tuple(prediction_results["symbol"]))} '
                                     f'AND date in {str(tuple(prediction_results["date"].astype(str)))}',
                                     con=POSTGRES_URL)

        results = prediction_results.merge(closing_prices, on=['symbol', 'date'])

        # we don't just buy anything the model classifies as a buy. We look for
        # a reasonable price to book value, and positive earnings.
        return results.loc[(results['predictions'] > .5)
                           & (results['pToBv'] > 0)
                           & (results['pToE'] > 0)
                           ]

    def predict(self):
        to_drop = ['date', 'date_prev']
        test_data = self.test_data.drop(axis=1, labels=to_drop)
        tf_test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data)
        self.test_data['predictions'] = self.model.predict(tf_test_dataset)
        return self.recommender(self.test_data[['symbol', 'predictions']].sort_values(by='predictions'))


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
