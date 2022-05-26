import pandas as pd
import logging
import os
from dotenv import load_dotenv
import functools
import numpy as np
from .transforms import interpolate


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cross_reference_database(iex_records: int, stock: str, db_table: str) -> int:
    """Only write new dataframe contents to the database. This will streamline the pipeline, and provide
    logging/metadata as to what is already in the database, and what the new data is that we're writing
    Parameters:
        iex_records: the number of records iex_cloud has for a given timeseries endpoint (E.G. Fundamentals).
        stock: the symbol in question.
        db_table: name of the table to cross references
    Returns:
        difference: the number of records we should pull to be up to date with iex offerings."""

    mysql_contents = pd.read_sql(f'SELECT COUNT(*) FROM {db_table} WHERE symbol="{stock}";', os.getenv('MYSQL_CONNECTION'))
    if iex_records != int(mysql_contents['COUNT(*)'][0]):
        difference = iex_records - int(mysql_contents['COUNT(*)'][0])
        logger.info(f' {difference} records being pulled from IEX and placed into {db_table} for {stock}')
    else:
        difference = 0
    return difference


def cast_as_dataframe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        request = func(*args, **kwargs)
        df = pd.DataFrame()
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
        return df
    return wrapper


def write_to_db(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """This function writes the contents of a dataframe to a sql table.
        The table name is the same as that of the function"""
        df = func(*args, **kwargs)
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        if kwargs['interpolate']:
            logger.info('Interpolating...')
            interpolate(df, date_col='reportDate',name=func.__name__)
        df.to_sql(func.__name__,
                  os.getenv('POSTGRES_CONNECTION'), if_exists='append', index=False)
        logger.info(f'Writing dataframe of shape {df.shape} to {func.__name__}')
    return wrapper






