import pandas as pd

from database import db_connector


def convert_mixed_epoch(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def convert_if_epoch(x):
        # everything is text in the database since dates and epoch time is intermixed (thanks IEX...),
        # this if statement makes sure date is not null, and checks to see if a '-' is in the entry
        # which indicates that its already in date format and no further work is needed.

        if isinstance(x, int) or (x and '-' not in x):
            x = int(x)
            if x > 1e11:  # Timestamp is in milliseconds
                return pd.to_datetime(x, unit='ms')
            elif x > 1e8:  # Timestamp is in seconds
                return pd.to_datetime(x, unit='s')
        return x

    df[column] = df[column].apply(convert_if_epoch)
    return df


def upsample_df(df, date_column, pivot=False, **kwargs):
    if pivot:
        df = df.pivot(index=date_column, columns=kwargs.get('columns'), values=kwargs.get('values'))

    else:
        # Set the date column as the index
        df = df.set_index(date_column)

    # Resample the DataFrame to daily frequency, then interpolate the missing values
    df = df.resample('D').asfreq()
    df = df.interpolate(method='linear')

    # Reset the index to move the date back to the columns
    df = df.reset_index()

    return df


if __name__ == '__main__':
    db_connector.connect()
    from config.metadata import tables

    for table, metadata in tables.items():
        df = db_connector.run_query(f'SELECT * FROM raw.{table}')
        df = df[metadata['columns']]  # only take certain columns.
        df.drop_duplicates(inplace=True)

        date_col = df.columns.to_list()[1]
        # cast date column as proper format. date col needs to always be 2nd
        df = convert_mixed_epoch(df, date_col)

        df.rename(inplace=True, columns={date_col: 'date'})

        if metadata['upsample']:
            df = upsample_df(df, date_col, **metadata['upsample_kwargs'])

        # create primary key
        df.insert(0, 'id', range(0, 0 + len(df)))

        # insert to db
        db_connector.insert_dataframe(df, name=table, index=False, if_exists='replace', schema='transform')

        # add pk to table.
        db_connector.modify_table(f'{table}', schema_name='transform', action="add_primary_key",
                                  column="id", constraint_name=f"{table}_pkey")
