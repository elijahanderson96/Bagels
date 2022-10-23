import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def test_model(self, path_to_file, symbol):
    model = tf.keras.models.load_model(path_to_file)
    data = pd.read_sql(f"""SELECT * FROM market.fundamentals
                        WHERE symbol = {symbol} ORDER BY reportdate ASC;""").rename(columns={'reportdate': 'date'})
    print(data)
    print(model.summary())
    data['date'] = pd.to_datetime(data['date']) + timedelta(days=91)  # offset 91 days.
    train_data = data.loc[data['date'] < datetime.now()]
    test_data = data.loc[data['date'] >= datetime.now()].sort_values(by=['symbol'])
    train_data = train_data.loc[train_data['symbol'].isin(test_data['symbol'])]
    idx = test_data.groupby('symbol')['date'].transform(max) == test_data['date']
    test_data = test_data[idx]
    sector = list(train_data.symbol.unique())
    logger.info(f'Training data is of shape {train_data.shape}')
    logger.info(f'Testing data is of shape {test_data.shape}')
    idx = train_data.groupby('symbol')['date'].transform(max) == train_data['date']
    validation_data = train_data[idx]
    train_data = train_data.loc[~idx]
    logger.info(f'Training data is of shape {train_data.shape}')
    logger.info(f'Validation data is of shape {test_data.shape}')
    labels = pd.read_sql(f"""SELECT date, close * shares_outstanding as marketcap, symbol
    FROM market.stock_prices WHERE symbol = {symbol};""")

    train_dates = train_data.pop('date')
    train_data = train_data._get_numeric_data()
    columns = train_data.columns.to_list()

    train_data = scaler.fit_transform(train_data)
    validation_data = scaler.transform(validation_data)
    test_data = scaler.transform(test_data)

    all_data = pd.concat([train_data,validation_data,test_data])


