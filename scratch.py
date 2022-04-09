import os
import requests
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()  # take environment variables from .env.

pd.options.display.max_rows = 20
pd.options.display.max_columns = 20


class Iex_Base:
    def __init__(self,production=False):
        if production:
            self.token = os.getenv('PRODUCTION_TOKEN')
            self.base_url = 'https://cloud.iexapis.com/'
        else:
            self.token = os.getenv('SANDBOX_TOKEN')
            self.base_url = 'https://sandbox.iexapis.com/'

        self.version = 'stable'


class TimeSeries(Iex_Base):
    def __init__(self, stock=None,endpoint=None, production=False):
        super().__init__(production)
        self.stock = stock
        self.endpoint = endpoint
        self.url = self.base_url + self.version + '/'

    def get_timeseries_inventory(self):
        """We explicitly declare URL here since it requires prod token and base_url"""
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series' + f'?token={os.getenv("PRODUCTION_TOKEN")}'
        print(url)
        inventory = requests.get(url=url)
        cast = inventory.json()
        df = pd.json_normalize(cast)
        print(df.columns.to_list())

    def get_fundamentals(self):
        pass

x = TimeSeries()
x.get_timeseries_inventory()

