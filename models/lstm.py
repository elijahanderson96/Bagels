import logging
from datetime import datetime
from datetime import timedelta
from functools import reduce
from hashlib import sha256
from time import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from database import db_connector
from one_time_sql_scripts.ingestion_fred import data_refresh

# Setting up logging
logging.basicConfig(level=logging.INFO)


def get_data_from_tables(table_names):
    """
    Fetch data from database tables
    Args:
    table_names (list): List of table names
    Returns:
    list: List of dataframes
    """
    logging.info('Fetching data from tables...')
    db_connector.connect()

    dataframes = [db_connector.run_query(f'SELECT * FROM fred_raw.{table}') for table in table_names]

    for i, df in enumerate(dataframes):
        dataframes[i] = df.rename(columns={'value': table_names[i]})

    logging.info('Data fetched successfully.')
    return dataframes


def align_dates(df_list, date_column, days=28):
    """
    Align dates across different dataframes in the list
    Args:
    df_list (list): List of dataframes
    date_column (str): Name of the column containing dates
    Returns:
    DataFrame: Merged DataFrame with aligned dates
    """
    logging.info('Aligning dates across dataframes...')
    target_max_date = datetime.now()

    for i, df in enumerate(df_list):
        df[date_column] = df[date_column].astype('datetime64[ns]')
        date_diff = target_max_date - df[date_column].max()
        df[date_column] = df[date_column] + date_diff
        df.set_index(date_column, inplace=True)
        df_list[i] = df

    df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), df_list)
    df_final.reset_index(inplace=True)
    df_final['date'] = df_final['date'].apply(lambda date: date.date())

    drop_columns = ['date', 'symbol'] if 'symbol' in df_final.columns else ['date']
    df_final.drop_duplicates(inplace=True, subset=drop_columns)
    logging.info('Date alignment completed.')
    df_final['date'] = df_final['date'] + timedelta(days=days)
    return df_final


def get_shift_amount(days=28):
    """Calculate the shift amount for weekdays."""
    shift_amount = days // 7 * 5 + min(days % 7, 5)
    return shift_amount


def get_labels(days=28):
    """
    Fetch labels from the database and generate binary labels
    Returns:
    DataFrame: DataFrame with binary labels and closing price at prediction
    """
    logging.info('Fetching and processing labels...')
    labels = db_connector.run_query(
        'SELECT SYMBOL, DATE, CLOSE '
        'FROM FRED_RAW.HISTORICAL_PRICES '
    )
    labels['date'] = labels['date'].apply(lambda date: date.date())
    labels.set_index('date', inplace=True)
    labels.sort_index(inplace=True)
    shift_amount = get_shift_amount(days)
    labels['future_price'] = labels.groupby('symbol')['close'].shift(-shift_amount)
    labels['price_change'] = labels['future_price'] - labels['close']
    labels['label'] = (labels['price_change'] > 0).astype(int)
    labels.rename(columns={'close': 'closing_price_at_prediction'}, inplace=True)
    logging.info('Label processing completed.')
    labels.reset_index(inplace=True)
    labels['date'] = labels['date'] + timedelta(days=days)
    return labels


def split_data(features_df, labels_df, backtest=False, cutoff_date=None, days=28):
    """
    Split data into training and prediction sets
    Args:
    features_df (DataFrame): DataFrame with features
    labels_df (DataFrame): DataFrame with labels
    backtest (bool): If true, cutoff_date must be set. Splits data into training and backtest sets.
    cutoff_date (str): The cutoff date for backtest. The format should be 'yyyy-mm-dd'.
    Returns:
    Tuple: train_data, test_data, and closing_price_at_prediction DataFrames
    """
    logging.info('Splitting data into training and prediction sets...')
    data_df = features_df.merge(labels_df, how='left', on=['date'])
    data_df.dropna(subset=['label'], inplace=True)
    data_df['date'] = pd.to_datetime(data_df['date'])

    if backtest:
        if cutoff_date is None:
            raise ValueError("When backtest=True, you must provide a valid cutoff_date.")

        cutoff_date = pd.to_datetime(cutoff_date)

        train_df = data_df[data_df['date'] <= cutoff_date]
        predict_df = data_df[data_df['date'] == cutoff_date + timedelta(days=days)]

    else:
        today = pd.to_datetime(datetime.now().date())
        train_df = data_df[data_df['date'] <= today]
        predict_df = data_df[data_df['date'] == today + timedelta(days=days - 1)]

    train_df = train_df.drop(columns=['future_price', 'price_change', 'closing_price_at_prediction'])
    train_df.drop_duplicates(inplace=True, subset=['symbol', 'date'])

    predict_df['date'] = pd.to_datetime(predict_df['date'])
    predict_df.drop_duplicates(inplace=True, subset=['symbol', 'date'])

    closing_price_at_prediction = predict_df[['symbol', 'date', 'closing_price_at_prediction']]
    predict_df = predict_df.drop(columns=['future_price', 'price_change', 'closing_price_at_prediction'])
    closing_price_at_prediction['date'] = pd.to_datetime(closing_price_at_prediction['date'])
    closing_price_at_prediction.drop_duplicates(inplace=True, subset=['symbol', 'date'])
    logging.info('Data split completed.')

    return train_df, predict_df, closing_price_at_prediction


def preprocess_data(df):
    """
    Preprocesses the data for LSTM
    Args:
        df (DataFrame): DataFrame to preprocess
        is_train (bool): True if the df is training data (includes labels)
    Returns:
        Tuple: Processed data, and labels if is_train=True
    """
    logging.info('Preprocessing data...')
    df = df.set_index('date')
    df.sort_index()
    # For training data, extract labels
    df.dropna(inplace=True)
    labels = df.pop('label')

    # Scale the features to be between 0 and 1
    scaler = MinMaxScaler()
    df = pd.get_dummies(df, columns=['symbol'])
    training_columns = df.columns
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Convert the dataframe into a 3D array (samples, timesteps, features) for LSTM
    data = np.expand_dims(df.values, axis=1)

    logging.info('Data preprocessing completed.')
    return data, labels, scaler, training_columns


def define_and_train_model(X_train, y_train):
    """
    Define and train LSTM model
    Args:
        X_train (ndarray): Training data
        y_train (Series): Training labels
    Returns:
        Sequential: Trained model
    """
    logging.info('Defining the model...')
    model = Sequential()
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(X_train.shape[2], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='accuracy', patience=50, restore_best_weights=True)

    # Define ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='accuracy', save_best_only=True, mode='max')

    # Fit the model
    logging.info('Training the model...')
    model.fit(
        X_train, y_train, epochs=100,
        callbacks=[early_stopping, model_checkpoint]
    )
    logging.info('Model training completed.')
    return model

def predict(model, predict_data, train_columns, scaler):
    """
    Generate predictions with the trained model
    Args:
    model (Sequential): Trained model
    predict_data (DataFrame): Data to predict
    train_columns (list): Column names from the training data
    scaler (MinMaxScaler): Fitted scaler for renormalization
    Returns:
    ndarray: Predictions
    """
    logging.info('Making predictions...')
    predict_data = pd.get_dummies(predict_data, columns=['symbol'])

    # Add missing columns
    missing_cols = set(train_columns) - set(predict_data.columns)

    for c in missing_cols:
        predict_data[c] = 0

    # Ensure the order of column in the test data is in the same order than in train set
    predict_data = predict_data[train_columns]

    predict_data = scaler.transform(predict_data)
    predict_data = np.expand_dims(predict_data, axis=1)

    predictions = model.predict(predict_data)
    logging.info('Predictions completed.')
    return predictions


def generate_model_id(model_name):
    """
    Generate a unique id for a given model name
    Args:
    model_name (str): Name of the model
    Returns:
    str: Unique ID for the model
    """
    current_time = str(time())
    model_id = sha256((model_name + current_time).encode()).hexdigest()
    return model_id


def store_predictions(predictions_df, model_id):
    """
    Store model predictions in the database
    Args:
    predictions_df (DataFrame): DataFrame with model predictions
    model_id (str): ID of the model used for the predictions
    """
    logging.info('Storing model predictions...')
    predictions_df['model_id'] = model_id
    kwargs = {'name': 'model_predictions', 'if_exists': 'append', 'schema': 'fred_raw', 'index': False}
    db_connector.insert_dataframe(predictions_df, **kwargs)
    logging.info('Model predictions stored successfully.')


def store_metrics(metrics, model_name):
    """
    Store model metrics in the database
    Args:
    metrics (dict): Dictionary with model metrics
    model_name (str): Name of the model
    """
    logging.info('Storing model metrics...')
    model_id = generate_model_id(model_name)
    metrics_df = pd.DataFrame(
        {
            'date': [datetime.now()],
            'model_id': [model_id],
            'model_name': [model_name],
            'accuracy': [metrics[1]],
            'loss': [metrics[0]]
        }
    )
    db_connector.insert_dataframe(metrics_df, name='model_metrics', if_exists='append', schema='fred_raw', index=False)
    logging.info('Model metrics stored successfully.')
    return model_id


def get_current_price(symbol):
    """
    Fetch the latest stock price for a given symbol

    Args:
        symbol (str): The stock symbol
    Returns:
        float: The current stock price
    """
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


def classify_stocks(predictions_df):
    """
    Classify stocks as overvalued or undervalued based on their current price and the predicted price

    Args:
        predictions_df (DataFrame): DataFrame with model predictions
    Returns:
        DataFrame: DataFrame with added 'classification' column
    """
    # Add new columns for current price and classification
    predictions_df['current_price'] = predictions_df['symbol'].apply(get_current_price)
    predictions_df['classification'] = 'fair'

    # Classify as overvalued or undervalued based on model prediction
    predictions_df.loc[(predictions_df['prediction'] <= .5) & (predictions_df['current_price'] > predictions_df[
        'closing_price_at_prediction']), 'classification'] = 'overvalued'

    predictions_df.loc[(predictions_df['prediction'] > .5) & (predictions_df['current_price'] < predictions_df[
        'closing_price_at_prediction']), 'classification'] = 'undervalued'

    return predictions_df


# Create up to date data
#data_refresh()

# Execute the process
tables = db_connector.run_query(
    '''SELECT TABLE_NAME 
       FROM INFORMATION_SCHEMA.TABLES 
       WHERE TABLE_SCHEMA = 'fred_raw' 
       AND TABLE_NAME NOT IN ('endpoints', 'historical_prices', 
       'model_metrics', 'model_predictions')'''
)['table_name']

dfs = get_data_from_tables(tables)
forecast_n_days_in_advance = 14

features_df = align_dates(dfs, 'date', days=forecast_n_days_in_advance)
labels_df = get_labels(days=forecast_n_days_in_advance)

# For backtesting
backtest = False
cutoff_date = '2023-02-22'

train_df, predict_df, closing_price_df = split_data(features_df,
                                                    labels_df,
                                                    backtest,
                                                    cutoff_date,
                                                    days=forecast_n_days_in_advance)

train_df.drop_duplicates(inplace=True, subset=['symbol', 'date'])

train_data, train_labels, scaler, columns = preprocess_data(train_df)

# Split training data into training and validation
#X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.001, shuffle=False)

# Preprocess data
#train_data, train_labels, scaler, columns = preprocess_data(train_df)

# Train the model
model = define_and_train_model(train_data, train_labels)

# Generate predictions
predictions = predict(model, predict_df, columns, scaler)

# Create a DataFrame with symbols, corresponding predictions, and closing price at prediction
result_df = pd.DataFrame(
    {
        'symbol': predict_df['symbol'].to_list(),
        'prediction': predictions.flatten()
    }
)

result_df = result_df.merge(closing_price_df, on='symbol', how='left')

result_df = classify_stocks(result_df)

# Fetch model metrics
#metrics = model.evaluate(X_val, y_val)

# Store the metrics
#model_id = store_metrics(metrics, 'incomplete_etf_model')

# Store the predictions
#store_predictions(result_df, model_id)
