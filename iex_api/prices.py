import pandas as pd
import os
import logging
from yfinance import download

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class UpdatePrices:
    def __init__(self, stock, shares_outstanding=None ,from_date=None):
        self.stock = stock
        self.shares_outstanding = shares_outstanding
        self.from_date = from_date
        self.engine = os.getenv('POSTGRES_CONNECTION')
        self.raw_financials_df = pd.read_sql(
                f'SELECT MIN("reportDate") '
                f"FROM fundamentals fm WHERE symbol='{self.stock}';",
                self.engine).astype('datetime64')
        # eff it, lets just download from our earliest fundamentals date everytime.
        self.prices = download(self.stock, start=self.raw_financials_df['min'][0]).reset_index()
        self.prices['symbol'] = self.stock
        self.prices['sharesOutstanding'] = self.shares_outstanding
        self.prices['marketCap'] = self.prices['Adj Close'] * self.shares_outstanding
        self.prices.to_sql('stock_prices', self.engine, if_exists='append', index=False)
        self.prices = pd.concat(
            [self.prices, (pd.read_sql('SELECT * FROM stock_prices', self.engine))]
        ).drop_duplicates(subset=['Date', 'symbol'])
        self.prices.to_sql('stock_prices', self.engine, if_exists='replace', index=False)
        logger.info(f'Inserted historical stock price data to Stock_Prices table for {self.stock.upper()}')
        logger.info(f'{self.stock.upper()} has {self.shares_outstanding} shares outstanding.')


if __name__ == '__main__':
    x = UpdatePrices(stock='BAC')
