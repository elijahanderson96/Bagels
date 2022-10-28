from config.configs import *
import pandas as pd
from helpers.helpers import parse_sql_file
QUERIES = parse_sql_file(full_path='/mnt/c/Users/Elijah/PycharmProjects/Bagels/sql/queries.sql')
SYMBOLS=pd.read_sql('''SELECT symbol FROM (SELECT symbol,
count(symbol)
FROM market.fundamental_valuations
WHERE subkey='ttm'
group by symbol) q
WHERE q.count > 5
ORDER BY symbol ASC;''',con=POSTGRES_URL)['symbol'].to_list()

