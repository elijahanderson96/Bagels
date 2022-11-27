import logging

import pandas as pd

from config.configs import *
from helpers.helpers import parse_sql_file

QUERIES = parse_sql_file(full_path='/mnt/c/Users/Elijah/PycharmProjects/Bagels/sql/queries.sql')
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

SYMBOLS = pd.read_sql('''SELECT symbol FROM (SELECT symbol,
count(symbol)
FROM market.fundamental_valuations
WHERE subkey='ttm'
group by symbol) q
WHERE q.count > 5
ORDER BY symbol ASC;''', con=POSTGRES_URL)['symbol'].to_list()

SMALLEST_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 1000000000 AND 2000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

SMALL_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 2000000000 AND 4000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

SMALL_TO_MID_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 4000000000 AND 10000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

MID_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 10000000000 AND 25000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

LARGE_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 25000000000 AND 100000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

MEGA_CAP_SYMBOLS = pd.read_sql(f"""
SELECT t.symbol FROM market.stock_prices t
inner join (select symbol, max(date) as max_date from market.stock_prices group by symbol) sp on sp.max_date = t.date
WHERE t.marketcap BETWEEN 100000000000 AND 900000000000000
AND t.symbol in {tuple(SYMBOLS)};
""", con=POSTGRES_URL)['symbol'].unique().tolist()

EXCLUDE_LIST = ['BHVN', 'BRK.A', 'CDR', 'CRD.B', 'RDUS', 'SAIL', 'DGLY', 'J', 'SPXC', 'AWI', 'AM', 'CLF',
                'ABNB', 'BEKE', 'BNTX', 'CIVI', 'COIN', 'CPNG', 'ELV', 'ET', 'HOOD', 'PARA', 'PDD',
                'RBLX', 'VTRS', 'WBD', 'U']

sym_mkcap_mappings = {
    'SMALLEST_CAP_SYMBOLS': SMALLEST_CAP_SYMBOLS,
    'SMALL_CAP': SMALL_CAP_SYMBOLS,
    'SMALL_TO_MID_CAP': SMALL_TO_MID_CAP_SYMBOLS,
    'MID_CAP': MID_CAP_SYMBOLS,
    'LARGE_CAP': LARGE_CAP_SYMBOLS,
    'MEGA_CAP': MEGA_CAP_SYMBOLS
}
