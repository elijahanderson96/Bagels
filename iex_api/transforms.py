import pandas as pd
from datetime import timedelta
import os
import numpy as np


def interpolate(df: pd.DataFrame, date_col: str = None, name='') -> pd.DataFrame:
    df = df.sort_values(by=['symbol', 'reportDate'])
    report_dates = df.pop('reportDate')
    symbols = df.pop('symbol')
    df = df._get_numeric_data()
    report_dates = pd.to_datetime(report_dates)
    final_df = pd.DataFrame()
    for i in range(len(report_dates)):
        print('interpolating')
        if i + 1 < len(report_dates):
            if symbols.iloc[i + 1] != symbols.iloc[i]:
                continue
            symbol = symbols.iloc[i]
            n_days = abs((report_dates.iloc[i + 1] - report_dates.iloc[i]).days)
            dates = pd.date_range((report_dates.iloc[i]),
                                  (report_dates.iloc[i + 1]), freq='d').strftime(
                "%Y-%m-%d").tolist()
            temp_values = {}
            for col in df.columns:
                if col != date_col:
                    tmp = []
                    for j in range(n_days + 1):
                        number = ((df[col].iloc[i + 1] - df[col].iloc[i]) / n_days * j) + df[col].iloc[i]
                        tmp.append(number)
                    temp_values.update({col: tmp})
            temp_values.update({'dates_interpolated': dates, 'symbol': symbol})
            tmp_df = pd.DataFrame(temp_values)
            final_df = pd.concat([final_df, tmp_df])

    final_df.drop_duplicates(subset=['dates_interpolated', 'symbol']
                             ).to_sql(f'interpolated_{name}',
                                      os.getenv('POSTGRES_CONNECTION'),
                                      if_exists='replace',
                                      index=False)
    return final_df
