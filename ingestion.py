import argparse
from datetime import datetime, timedelta
from time import sleep
from traceback import print_exc

import pandas as pd
import requests

from database import db_connector


def retrieve_data(base_url, start_date, end_date, chunk_size):
    data = []
    current_date = start_date

    while current_date < end_date:
        next_date = min(current_date + timedelta(days=chunk_size), end_date)
        print(f'Grabbing data from {current_date} to {next_date}.')
        response = requests.get(base_url, params={'token': 'pk_ae9aaede092843cda2c7fb0872792c89',
                                                  'from': current_date.strftime('%Y-%m-%d'),
                                                  'to': next_date.strftime('%Y-%m-%d')})
        response_json = response.json()

        if response_json:
            data.extend(response_json)

        current_date = next_date + timedelta(days=1)
        sleep(.1)

    df = pd.DataFrame(data)
    return df.where(pd.notnull(df), None)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Ingest a dataset based on start_date, end_date and endpoint name')
    parser.add_argument('-start', '--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('-end', '--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('-endpoint', '--endpoint_name', type=str, help='Endpoint name')

    args = parser.parse_args()

    if not args.start_date or not args.end_date or not args.endpoint_name:
        raise ValueError('Need to specify a start date, end date, and which endpoint you are ingesting.')

    # Convert dates to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')


    def fetch_data(url, chunk_size, start_date, end_date):
        print(f'Grabbing data for {url}.')
        try:
            df = retrieve_data(url, start_date, end_date, chunk_size)
            kwargs = {'name': f'{args.endpoint_name.lower()}', 'index': False, 'schema': 'raw', 'if_exists': 'append'}
            print('Inserting data.')
            db_connector.insert_dataframe(df, **kwargs)

        except ValueError:
            print(print_exc())


    # usage
    base_url = f'https://cloud.iexapis.com/v1/data/CORE/{args.endpoint_name.upper()}'

    if args.endpoint_name.upper() == 'HISTORICAL_PRICES':
        stocks = ['SCHD', 'FNDX', 'SDY', 'VOO', 'VOOG', 'VOOV', 'VV', 'VUG', 'VTV', 'MGC']
        [fetch_data(base_url + f'/{symbol}',
                    chunk_size=3650,
                    start_date=start_date,
                    end_date=end_date) for symbol in stocks]

    else:
        fetch_data(base_url, chunk_size=30, start_date=start_date, end_date=end_date)
