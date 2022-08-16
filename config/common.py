from config.configs import *
import pandas as pd

MARKET_DATA = pd.read_csv('./app_data/total_market.csv')
MARKET_DATA['Symbol'] = MARKET_DATA['Symbol'].astype('str')
SYMBOLS = [sym for sym in MARKET_DATA['Symbol'] if '^' not in sym]


def insert_new_entries(table, data_to_insert):
    q = 'INSERT INTO tbl2 VALUES (values,here) ON CONFLICT (column,to_reference) DO NOTHING;'
    df = pd.read_sql(f"SELECT * FROM {table}", con=POSTGRES_URL)
    dfs = pd.concat([df, data_to_insert], ignore_index=True)
    dfs.drop_duplicates(inplace=True, keep='last')
    df.to_sql(table, con=POSTGRES_URL, schema='market', if_exists='append')
