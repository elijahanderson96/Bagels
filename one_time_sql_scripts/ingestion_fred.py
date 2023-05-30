from time import sleep

import pandas as pd
import yfinance
from fredapi import Fred

from database import db_connector

fred = Fred(api_key='7f54d62f0a53c2b106b903fc80ecace1 ')

endpoints = {'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average',
             'DFF': 'Federal Funds Effective Rate',
             'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average in the United States',
             'BAMLH0A0HYM2': 'ICE BofA US High Yield Index Option-Adjusted Spread',
             'T10Y3M': '10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity',
             'UNRATE': 'Unemployment Rate',
             'GDP': 'Gross Domestic Product',
             'CSUSHPISA': 'S&P/Case-Shiller U.S. National Home Price Index',
             'DGS10': 'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, '
                      'Quoted on an Investment Basis',
             'GDPC1': 'Real Gross Domestic Product',
             'T10YIE': '10-Year Breakeven Inflation Rate',
             'RRPONTSYD': 'Overnight Reverse Repurchase Agreements: '
                          'Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations',
             'MSPUS': 'Median Sales Price of Houses Sold for the United States',
             'T5YIE': '5-Year Breakeven Inflation Rate',
             'DFII10': 'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity'
                       'Quoted on an Investment Basis, Inflation-Indexed',
             'DPSACBW027SBOG': 'Deposits, All Commercial Banks',
             'GFDEGDQ188S': 'Federal Debt: Total Public Debt as Percent of Gross Domestic Product',
             'CORESTICKM159SFRBATL': 'Sticky Price Consumer Price Index less Food and Energy',
             'WTREGEN':
                 'Liabilities and Capital: Liabilities: Deposits with F.R. Banks, '
                 'Other Than Reserve Balances: U.S. Treasury, General Account: Week Average',
             'PSAVERT': 'Personal Saving Rate',
             'ASPUS': 'Average Sales Price of Houses Sold for the United States',
             'GFDEBTN': 'Federal Debt: Total Public Debt',
             'CIVPART': 'Labor Force Participation Rate'
             }

metadata = pd.DataFrame(data={'id': range(0, len(endpoints)),
                              'endpoint': endpoints.keys(), 'value': endpoints.values()})
db_connector.insert_dataframe(metadata, schema='fred_raw', name='endpoints', if_exists='replace', index=False)

for endpoint in endpoints.keys():
    sleep(.25)
    print(f'Grabbing {endpoint}.')
    series = pd.DataFrame(fred.get_series(endpoint, observation_start='2000-01-01'))
    # Resample the DataFrame to daily frequency, then interpolate the missing values
    series = series.resample('D').asfreq()
    series = series.interpolate(method='linear')

    # Reset the index to move the date back to the columns
    series = series.reset_index()
    series.rename(columns={0: 'value', 'index': 'date'}, inplace=True)
    series['endpoint_id'] = metadata.loc[metadata['endpoint'] == endpoint]['id'].squeeze()
    db_connector.insert_dataframe(series, schema='fred_raw', name=endpoint.lower(), if_exists='replace', index=False)

etfs = ['VT', 'VTI', 'VYM', 'SPHD', 'IWV', 'ITOT', 'ACWI', 'SCHB', 'FNDB', 'VXUS', 'VTHR', 'DIA', 'RSP',
        'IOO', 'IVV', 'SPY', 'VOO', 'SHE', 'VOO', 'IWM', 'OEF', 'QQQ', 'CVY', 'RPG', 'RPV', 'IWB', 'IWF',
        'IWD', 'IVV', 'IVW', 'IVE', 'PKW', 'PRF', 'SPLV', 'SCHX', 'SCHG', 'SCHV', 'SCHD', 'FNDX', 'SDY',
        'VOO', 'VOOG', 'VOOV', 'VV', 'VUG', 'VTV', 'MGC']

dfs = []

for etf in etfs:
    sleep(.25)
    print(f'Grabbing {etf}.')
    df = yfinance.download(etf, start='2000-01-01', end='2023-06-01')
    df['symbol'] = etf
    df.reset_index(inplace=True)
    [df.rename(columns={col: col.replace(' ', '_').lower()}, inplace=True) for col in df.columns.to_list()]
    dfs.append(df)

df = pd.concat(dfs)
db_connector.insert_dataframe(df, schema='fred_raw', name='historical_prices', if_exists='replace', index=False)
