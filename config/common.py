from config.configs import *
import pandas as pd

# MARKET_DATA = pd.read_csv('./app_data/total_market.csv')
# MARKET_DATA['Symbol'] = MARKET_DATA['Symbol'].astype('str')
# SYMBOLS = [sym for sym in MARKET_DATA['Symbol'] if '^' not in sym]

# these are the symbols we currently have REAL PRODUCTION DATA FOR
SYMBOLS = ['JNJ', 'JPM', 'BAC', 'XOM', 'CVX', 'LLY', 'ABBV', 'KO', 'HD', 'PG', 'PEP', 'MRK', 'AVGO', 'MCD']
