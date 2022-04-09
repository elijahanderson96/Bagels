import os
import requests
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()  # take environment variables from .env.

pd.options.display.max_columns = 10
pd.options.display.max_rows = 150


class Iex_Base:
    def __init__(self, production=False):
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
        self.url = self.base_url + self.version + '/time-series/'

    def get_timeseries_inventory(self):
        """We explicitly declare URL here since it requires prod token and base_url"""
        columns_of_interest = ('id', 'description', 'weight', 'updated', 'providerName', 'provider')
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series' + f'?token={os.getenv("PRODUCTION_TOKEN")}'
        temp = {}
        df = pd.DataFrame(columns=columns_of_interest)
        inventory = requests.get(url=url)
        content = inventory.json()
        for obj in content:                 #Content is a list of dictionaries comprising the different time series
            for key, value in obj.items():  #For each key,value, add to dataframe if key is in columns of interest
                if key in columns_of_interest:
                    temp.update({key: [value]})
            temp_df = pd.DataFrame(temp)
            df = pd.concat([df, temp_df])
        return df

    def get_fundamentals(self, stock, subkey='ttm', last=5):
        assert subkey in ['ttm', 'quarterly'], 'Subkey must be ttm or quarterly'
        url = self.url + f'/FUNDAMENTALS/{stock}/{subkey}?last={last}&token={self.token}'
        r = requests.get(url)
        content = r.json()
        df = pd.DataFrame()
        for entry in content:
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
            print(df)
        return df


x = TimeSeries()
#x.get_timeseries_inventory()
x.get_fundamentals('CSIQ',subkey='ttm')

