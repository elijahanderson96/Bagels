import logging
from yfinance import download
from config.common import *
from data.iex import Iex

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Prices(Iex):
    """This class is responsible for constructing the stocks and prices tables.
    The prices table is updated daily. THe stocks table contains metadata regarding
    what sector a stock is in."""

    def __init__(self):
        super().__init__()


    def _assign_stock_to_sector(self):
        sectors = MARKET_DATA.loc[MARKET_DATA['Sector'].notnull()]['Sector']
        stock_sector_mapping = pd.DataFrame(data={'stock': SYMBOLS, 'sector': sectors})
        return stock_sector_mapping

    def fetch_stock_price(self):
        from time import sleep
        dfs = []
        for symbol in SYMBOLS[0:5]:
            sleep(1)
            df = download(symbol, group_by='ticker', auto_adjust=True, threads=True)  # ,period='1d')
            df['symbol'] = symbol
            df['shares_outstanding'] = super()._shares_outstanding(symbol)
            df['marketCap'] = df['Close'] * df['shares_outstanding']
            df.columns = map(str.lower, df.columns)  # make columns lowercase
            dfs.append(df)
        return pd.concat(dfs)

    def update_db(self):
        prices_matrix = self.fetch_stock_price()
        prices_matrix.to_sql('stock_prices', con=POSTGRES_URL, schema='market', if_exists='replace')
