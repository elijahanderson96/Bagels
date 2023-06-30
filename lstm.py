import logging
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Dict
from typing import List
from typing import Tuple

from database import db_connector
from one_time_sql_scripts.ingestion_fred import etfs

# Setting up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_data_from_tables(table_names: List[str]) -> List[pd.DataFrame]:
    """
    Fetch data from database tables

    Args:
        table_names (List[str]): List of table names

    Returns:
        List[pd.DataFrame]: List of dataframes
    """
    logging.info('Fetching data from tables...')
    db_connector.connect()
    try:
        dfs = [db_connector.run_query(f'SELECT * FROM transform.{table}')
               for table in table_names]
        dfs = [df.drop(axis=1, labels=['id']) for df in dfs]
    except Exception as e:
        logging.error(f'Failed to fetch data from tables due to {e}.')
        raise
    else:
        logging.info('Data fetched successfully.')
        return dfs


def align_dates(df_list: List[pd.DataFrame], date_column: str) -> pd.DataFrame:
    """
    Align dates across different dataframes in the list

    Args:
        df_list (List[pd.DataFrame]): List of dataframes
        date_column (str): Name of the column containing dates

    Returns:
        pd.DataFrame: Merged DataFrame with aligned dates
    """
    logging.info('Aligning dates across dataframes...')
    target_max_date = datetime.now()
    try:
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
    except Exception as e:
        logging.error(f'Failed to align dates due to {e}.')
        raise
    else:
        logging.info('Date alignment completed.')
        return df_final


def get_labels() -> pd.DataFrame:
    """
    Fetch labels from the database and generate binary labels

    Returns:
        pd.DataFrame: DataFrame with binary labels and closing price at prediction
    """
    logging.info('Fetching and processing labels...')
    try:
        labels_df = db_connector.run_query(
            'SELECT SYMBOL, DATE, CLOSE'
            ' FROM TRANSFORM.HISTORICAL_PRICES '
            f"WHERE SYMBOL IN {tuple(etfs)}"
        )
        labels_df['date'] = labels_df['date'].apply(lambda date: date.date())
        labels_df.sort_values(['symbol', 'date'], inplace=True)
        labels_df['future_price'] = labels_df.groupby('symbol')['close'].shift(-77)
        labels_df['price_change'] = labels_df['future_price'] - labels_df['close']
        labels_df['label'] = (labels_df['price_change'] > 0).astype(int)
        labels_df.rename(columns={'close': 'closing_price_at_prediction'}, inplace=True)
    except Exception as e:
        logging.error(f'Failed to fetch and process labels due to {e}.')
        raise
    else:
        logging.info('Label processing completed.')
        return labels_df


def split_data(features_df: pd.DataFrame, labels_df: pd.DataFrame, merge_on_date: bool = True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and prediction sets

    Args:
        features_df (pd.DataFrame): DataFrame with features
        labels_df (pd.DataFrame): DataFrame with labels
        merge_on_date (bool): Whether to merge on date. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_data, test_data, and closing_price_at_prediction
        DataFrames
    """
    logging.info('Splitting data into training and prediction sets...')
    merge_on = ['date', 'symbol'] if merge_on_date else ['date']
    try:
        data_df = features_df.merge(labels_df, how='left', on=merge_on)

        data_df.dropna(subset=['label'], inplace=True)
        data_df['date'] = data_df['date'] + timedelta(days=77)

        today = datetime.now().date()

        train_df = data_df[data_df['date'] <= today]
        predict_df = data_df[data_df['date'] > today]

        train_df.drop(columns=['future_price', 'price_change'], inplace=True)
        closing_price_at_prediction = predict_df[['symbol', 'date', 'closing_price_at_prediction']]

        predict_df.drop(columns=['closing_price_at_prediction'], inplace=True)
        closing_price_at_prediction.drop_duplicates(subset=['symbol', 'date'], inplace=True)
    except Exception as e:
        logging.error(f'Failed to split data due to {e}.')
        raise
    else:
        logging.info('Data split completed.')
        return train_df, predict_df, closing_price_at_prediction


# ... continue with the rest of the functions ...


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, MinMaxScaler, List[str]]:
    """
    Preprocesses the data for LSTM

    Args:
        df (pd.DataFrame): DataFrame to preprocess

    Returns:
        Tuple[np.ndarray, pd.Series, MinMaxScaler, List[str]]: Processed data, labels, fitted scaler, training column
        names
    """
    logging.info('Preprocessing data...')
    try:
        df = df.set_index('date')
        df.sort_index()
        df.dropna(inplace=True)
        labels = df.pop('label')

        # Scale the features to be between 0 and 1
        scaler = MinMaxScaler()
        df = pd.get_dummies(df, columns=['symbol'])
        training_columns = df.columns
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Convert the dataframe into a 3D array (samples, timesteps, features) for LSTM
        data = np.expand_dims(df.values, axis=1)
    except Exception as e:
        logging.error(f'Failed to preprocess data due to {e}.')
        raise
    else:
        logging.info('Data preprocessing completed.')
        return data, labels, scaler, training_columns


def define_and_train_model(
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
) -> Sequential:
    """
    Define and train LSTM model

    Args:
        X_train (np.ndarray): Training data
        y_train (pd.Series): Training labels
        X_test (np.ndarray): Test data
        y_test (pd.Series): Test labels

    Returns:
        Sequential: Trained model
    """
    logging.info('Defining the LSTM model...')
    model = Sequential()
    model.add(
        LSTM(
            X_train.shape[2] // 2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=True
        )
    )
    model.add(
        LSTM(
            X_train.shape[2] // 4, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=True
        )
    )
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Fit the model
    logging.info('Training the model...')
    try:
        model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
    except Exception as e:
        logging.error(f'Failed to train the model due to {e}.')
        raise
    else:
        logging.info('Model training completed.')
        return model


def predict(
        model: Sequential,
        predict_data: pd.DataFrame,
        train_columns: List[str],
        scaler: MinMaxScaler) -> np.ndarray:
    """
    Generate predictions with the trained model

    Args:
        model (Sequential): Trained model
        predict_data (pd.DataFrame): Data to predict
        train_columns (List[str]): Column names from the training data
        scaler (MinMaxScaler): Fitted scaler for renormalization

    Returns:
        np.ndarray: Predictions
    """
    logging.info('Making predictions...')
    try:
        missing_cols = set(train_columns) - set(predict_data.columns)

        for c in missing_cols:
            predict_data[c] = 0

        # Ensure the order of column in the test data is in the same order than in train set
        predict_data = predict_data[train_columns]

        predict_data = np.expand_dims(scaler.transform(predict_data), axis=1)
        y_pred = model.predict(predict_data)
    except Exception as e:
        logging.error(f'Failed to make predictions due to {e}.')
        raise
    else:
        logging.info('Predictions made successfully.')
        return y_pred


# ... continue with the rest of the functions ...


def store_predictions(predictions_df: pd.DataFrame) -> None:
    """
    Store model predictions in the database.

    Args:
        predictions_df (pd.DataFrame): DataFrame with model predictions
    """
    logging.info('Storing model predictions...')
    try:
        db_connector.insert_dataframe(predictions_df, 'model_predictions')
    except Exception as e:
        logging.error(f'Failed to store model predictions due to {e}.')
        raise
    else:
        logging.info('Model predictions stored successfully.')


def store_metrics(metrics: Dict[str, float], model_name: str) -> None:
    """
    Store model metrics in the database

    Args:
        metrics (dict): Dictionary with model metrics
        model_name (str): Name of the model
    """
    logging.info('Storing model metrics...')
    try:
        metrics_df = pd.DataFrame(
            {
                'date': [datetime.now()],
                'model_name': [model_name],
                'accuracy': [metrics['accuracy']],
                'loss': [metrics['loss']]
            }
        )
        db_connector.insert_dataframe(metrics_df, 'model_metrics')
    except Exception as e:
        logging.error(f'Failed to store model metrics due to {e}.')
        raise
    else:
        logging.info('Model metrics stored successfully.')


# Execute the process
table_names = ['economic', 'energy', 'treasury']
dfs = get_data_from_tables(table_names)
features_df = align_dates(dfs, 'date')
labels_df = get_labels()
train_df, predict_df, closing_price_df = split_data(features_df, labels_df, merge_on_date=False)

train_data, train_labels, scaler, columns = preprocess_data(train_df)
# Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2)
# Train the model
model = define_and_train_model(X_train, y_train, X_val, y_val)

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
result_df.drop_duplicates(subset=['symbol', 'date'], inplace=True)
# Store the predictions
store_predictions(result_df)

# Fetch model metrics
metrics = model.evaluate(X_val, y_val)

# Store the metrics
store_metrics(metrics, 'my_lstm_model')
