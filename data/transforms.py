import logging
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, cpu_count
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FeaturePrepper:
    """Takes data from any endpoint and transform it. We impute row data such that
    any zeros are replaced by the average value of the row, and interpolate the data such
    that there's data for every day. It is a linear interpolation."""

    def __init__(self):
        self.data = None # fundamentals data
        self.macro_data = None
        self.chunked_data = None
        self.labeled_data = None
        self.symbols = None
        self.report_dates = None

    @staticmethod
    def interpolate(df_chunk):
        assert df_chunk['symbol'].iloc[0] == df_chunk['symbol'].iloc[1], 'symbols do not match'
        n_days = abs((df_chunk['date'].iloc[0] - df_chunk['date'].iloc[1]).days)
        dates = pd.date_range(df_chunk['date'].iloc[0], df_chunk['date'].iloc[1], freq='d')
        temp_values = {}
        for col in df_chunk.columns:
            if col not in ('date', 'symbol'):
                tmp = []
                for j in range(n_days + 1):
                    number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                    tmp.append(number)
                temp_values.update({col: tmp})
        temp_values.update({'date': dates, 'symbol': df_chunk['symbol'].iloc[0]})
        return pd.DataFrame(temp_values)

    @staticmethod
    def interpolate_macro_data(df_chunk):
        n_days = abs((df_chunk['date'].iloc[0] - df_chunk['date'].iloc[1]).days)
        dates = pd.date_range(df_chunk['date'].iloc[0], df_chunk['date'].iloc[1], freq='d')
        temp_values = {}
        for col in df_chunk.columns:
            if col not in ('date'):
                tmp = []
                for j in range(n_days + 1):
                    number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                    tmp.append(number)
                temp_values.update({col: tmp})
        temp_values['date'] = dates
        return pd.DataFrame(temp_values)

    def preprocess(self, interpolate=False):
        """Sorts the values by symbols (if multiple) and report dates. Pops the report dates (after adding 91 days)
        and symbols into a class attribute, and returns all numeric data from the original data"""
        self.data.fillna(0, inplace=True)
        self.data.sort_values(by=['symbol', 'date'], inplace=True)
        logger.info('Dropping any duplicate entries based on symbol and report date.')
        self.data.drop_duplicates(inplace=True, subset=['symbol', 'date'])  # add symbol
        self.fetch_numeric_data()
        if not interpolate: return
        logger.info('Chunking the dataframe to expedite interpolation')
        self.chunked_data = [self.data.iloc[i:i + 2] for i in range(len(self.data) - 1)
                                 if self.data['symbol'].iloc[i] == self.data['symbol'].iloc[i + 1]]
        return self.data

    def fetch_numeric_data(self):
        """Save report dates and symbols, drop non-numeric """
        report_dates = self.data['date'].to_list()
        symbols = self.data['symbol'].to_list()
        self.data = self.data._get_numeric_data()
        self.data['date'] = report_dates
        self.data['symbol'] = symbols

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df

    def transform(self, data, interpolate=False):
        """

        Args:
            data: Dataframe to prepare as input for the model.
            interpolate: Whether to linearize data that is not on a day-to-day basis

        Returns:

        """
        self.data = data
        self.preprocess(interpolate=interpolate)
        if not interpolate:
            logger.info('Data is NOT being interpolated.')
            return
        logger.info(f'Provisioning {cpu_count() - 1} cpus for transformation')
        with Pool(cpu_count() - 1) as p:
            dfs = p.map(self.interpolate, self.chunked_data)
            df = pd.concat(dfs, ignore_index=True)
            # df = self._impute_row_data(df)
        logger.info(f'Interpolation for {", ".join(df["symbol"].unique())} complete')
        return df


    def transform_macro_data(self, macro_data: list):
        """Sometimes we can't join a quarterly report on data that comes out weekly so
        we will linearize and approximate to allow for join across all dates"""
        dfs = []
        for df in macro_data:
            df['date'] = df['date'].astype('datetime64[ns]')
            df.drop_duplicates(subset=['date'],inplace=True)
            df.sort_values(inplace=True,by='date')
            chunked_data = [df.iloc[i:i + 2] for i in range(len(df) - 1)]
            df_interpolated = [self.interpolate_macro_data(chunk) for chunk in chunked_data]
            df_interpolated = pd.concat(df_interpolated, ignore_index=True)
            df_interpolated['date'] = pd.to_datetime(df_interpolated['date']) + timedelta(days=91)
            dfs.append(df_interpolated)
        logger.info(f'Interpolation for macrodata complete')
        return dfs

