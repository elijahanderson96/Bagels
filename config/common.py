from config.configs import *
import pandas as pd
from helpers.helpers import parse_sql_file
QUERIES = parse_sql_file(full_path='/mnt/c/Users/Elijah/PycharmProjects/Bagels/sql/queries.sql')

SYMBOLS=pd.read_sql('''SELECT *
    FROM (SELECT DISTINCT symbol
          FROM market.fundamental_valuations
         ) t
    ORDER BY random()
    LIMIT 10;''',con=POSTGRES_URL)['symbol'].to_list()
