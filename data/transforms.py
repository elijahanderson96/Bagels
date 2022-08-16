import pandas as pd
from config.configs import *
import numpy as np
from datetime import timedelta


class Transformer:
    def __init__(self):
        pass

    def interpolate(self, df: pd.DataFrame, date_col: str = 'reportdate', table_name='fundamentals') -> pd.DataFrame:

        """

        Args:
            df:
            date_col:
            name:

        Returns:

        """

        df = df.sort_values(by=['symbol', 'reportdate'])
        report_dates = df.pop('reportdate')
        symbols = df.pop('symbol')
        df = df._get_numeric_data()
        report_dates = pd.to_datetime(report_dates)
        final_df = pd.DataFrame()
        for i in range(len(report_dates)):
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

        final_df.drop_duplicates(subset=['dates_interpolated', 'symbol']) \
            .to_sql(f'interpolated_{table_name}',
                    POSTGRES_URL,
                    if_exists='append',
                    schema='market',
                    index=False)

        return final_df


    def create_feature_matrix(self):
        features = pd.read_sql()
        labels = pd.read_sql()
        # one hot encode somewhere here
        labeled_dataset = pd.merge()
        return labeled_dataset


df = pd.read_sql("SELECT * FROM market.fundamentals_imputed", con=POSTGRES_URL)

transformer = Transformer()
df_interpolated = transformer.interpolate(df)

# get max date of fundamentals table for each symbol and max date of interpolated table.
# if they dont match, find the entry that matches max and interpolate.
