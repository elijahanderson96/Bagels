import pandas as pd
import numpy as np
from multiprocessing import Pool


def impute_row_data(df):
    df.replace(0, np.nan, inplace=True)
    m = df.mean(axis=1)
    for i, col in enumerate(df):
        df.iloc[:, i] = df.iloc[:, i].fillna(m)
    return df


def interpolate(df_chunk):
    df_chunk['reportdate'] = df_chunk['reportdate'].astype('datetime64')
    n_days = abs((df_chunk['reportdate'].iloc[0] - df_chunk['reportdate'].iloc[1]).days)
    dates = pd.date_range(df_chunk['reportdate'].iloc[0], df_chunk['reportdate'].iloc[1], freq='d')
    temp_values = {}
    for col in df_chunk.columns:
        if col not in ('reportdate', 'symbol'):
            tmp = []
            for j in range(n_days + 1):
                number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                tmp.append(number)
            temp_values.update({col: tmp})
    temp_values.update({'date': dates, 'symbol': df_chunk['symbol'].iloc[0]})
    return pd.DataFrame(temp_values)


if __name__ == '__main__':
    df.sort_values(by=['symbol', 'reportdate'], inplace=True)
    df_chunked = [df.iloc[i:i + 2] for i in range(len(df) - 1)
                  if df['symbol'].iloc[i] == df['symbol'].iloc[i + 1]]
    with Pool(2) as p:
        x = p.map(interpolate, df_chunked)
        df = pd.concat(x, ignore_index=True)

print(df)
# for chunk in df_chunked:
# dfs = interpolate(chunk)
# print(dfs)
