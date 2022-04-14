import pandas as pd
import logging
import os
from dotenv import load_dotenv

load_dotenv

logging.basicConfig(filename='db.log', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
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
    def wrapper(*args, **kwargs):
        request = func(*args, **kwargs)
        df = pd.DataFrame()
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
        return df
    return wrapper





