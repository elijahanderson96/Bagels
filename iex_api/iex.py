import os
import requests
from dotenv import load_dotenv
import pandas as pd
from decorators import cast_as_dataframe, write_to_db
import logging
from datetime import date
from dateutil.relativedelta import relativedelta
from config import treasury_endpoints

load_dotenv()  # take environment variables from .env.


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
    def __init__(self, production=False):
        super().__init__(production)
        self.url = self.base_url + self.version + '/time-series/'

    @staticmethod
    @write_to_db
    @cast_as_dataframe
    def timeseries_inventory():
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series'
        r = requests.get(url=url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return r

    @staticmethod
    @cast_as_dataframe
    def timeseries_metadata(time_series_id=None, stock=None,
                            subkey='ttm'):  # TODO exception handling for incorrect time-series ids and stocks
        """See how many records IEX has to offer for a given time series for a given stock. Passing None
        for both time_series_id and stock gives you all the metadata (all time series for all stocks).
            Parameters:
                time_series_id: The time series endpoint. See get_time_series_inventory for available endpoints
                stock: the symbol in question.
                subkey: a filter parameter for the type of content returned. We'll stick with TTM as default and not annual
                since usually its redundant information.

            Returns:
                r: JSON response as a dataframe (thanks to the decorator)."""
        assert subkey in ['ttm', 'quarterly', 'annual'], 'Subkey must be ttm,annual,or quarterly'
        url = f'https://cloud.iexapis.com/stable/metadata/time-series/'
        if time_series_id and stock and not subkey:
            url += f'{time_series_id}/{stock}'
        elif time_series_id and stock and subkey:
            print('should be here')
            url += f'{time_series_id}/{stock}/{subkey}'
        elif not time_series_id and stock:
            url += f'*/{stock}'
        elif time_series_id and not stock:
            url += f'{time_series_id}'
        r = requests.get(url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return r

    # Fix the index of this dataframe
    @write_to_db
    @cast_as_dataframe
    def fundamentals(self, stock, subkey='ttm', last=5):
        assert subkey in ['ttm', 'quarterly'], 'Subkey must be ttm or quarterly'
        url = self.url + f'/FUNDAMENTALS/{stock}/{subkey}'
        print(url)
        r = requests.get(url, params={'last': last, 'token': self.token})
        return r

    @write_to_db
    @cast_as_dataframe
    def treasury_rates(self, symbol, last=5):
        """Only doing 30 year for now, id is dgs30"""
        assert symbol in ['dgs30'], 'Only considering 30 year treasure rates at this time'
        last_n_years = date.today() - relativedelta(years=last)
        print(last_n_years)
        url = self.url + f'TREASURY/{symbol}'
        print(url)
        r = requests.get(url, params={'token': self.token, 'from': last_n_years})
        return r


# Pipeline calls this file as bash operator to automatically write all our time series endpoints
# To the database and not make get requests on data we already have, and not write duplicates
# to the database.
if __name__ == '__main__':
    DataGetter = TimeSeries()
    DataGetter.fundamentals('C')
   # DataGetter.treasury_rates('dgs30')

# When I call get_fundamentals I want a detailed report on what got stored in the mysql database.
# This function should write to an appropriate table (Fundamentals) and it should tell me how many new entries
# It inserted, for starters. Other useful info can be included here as well.
# print(x.get_fundamentals('CSIQ'))
