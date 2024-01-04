import logging
import os
from datetime import datetime
from datetime import timedelta
from functools import reduce
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from database.database import db_connector
from scripts.ingestion_fred import load_config

# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatasetBuilder:
    def __init__(
            self, table_names: List[str], etf_symbol: str, forecast_n_days_ahead: int = 14, sequence_length: int = 14
    ):
        self.table_names = table_names
        self.etf_symbol = etf_symbol
        self.forecast_n_days_ahead = forecast_n_days_ahead
        self.sequence_length = sequence_length

    def _get_data_from_tables(self) -> List[pd.DataFrame]:
        logging.info("Fetching data from tables...")
        try:
            dfs = [
                db_connector.run_query(
                    f"SELECT * FROM {self.etf_symbol.lower()}.{table}"
                )
                for table in self.table_names
            ]
            for table, df in dict(zip(self.table_names, dfs)).items():
                df.rename(columns={"value": table}, inplace=True)
        except Exception as e:
            logging.error(f"Failed to fetch data from tables due to {e}.")
            raise
        else:
            logging.info("Data fetched successfully.")
            return dfs

    def _align_dates(
            self, df_list: List[pd.DataFrame], date_column: str
    ) -> pd.DataFrame:
        logging.info("Aligning dates across dataframes...")
        try:
            for i, df in enumerate(df_list):
                df[date_column] = df[date_column].astype("datetime64[ns]")
                df.set_index(date_column, inplace=True)
                df_list[i] = df

            df_final = reduce(
                lambda left, right: pd.merge(
                    left, right, left_index=True, right_index=True, how="outer"
                ),
                df_list,
            )
            df_final.reset_index(inplace=True)
            drop_columns = (
                ["date", "symbol"] if "symbol" in df_final.columns else ["date"]
            )
            df_final.drop_duplicates(inplace=True, subset=drop_columns)
        except Exception as e:
            logging.error(f"Failed to align dates due to {e}.")
            raise
        else:
            logging.info("Date alignment completed.")
            return df_final.dropna()

    def _get_labels(self) -> pd.DataFrame:
        logging.info("Fetching and processing labels...")
        try:
            labels_df = db_connector.run_query(
                f"SELECT date, close FROM {self.etf_symbol.lower()}.{self.etf_symbol}_historical_prices WHERE "
                f"symbol='{self.etf_symbol}'"
            )
            labels_df.sort_values("date", inplace=True)
        except Exception as e:
            logging.error(f"Failed to fetch and process labels due to {e}.")
            raise
        else:
            logging.info("Label processing completed.")
            return labels_df

    def _split_data(
            self,
            features_df: pd.DataFrame,
            labels_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info("Splitting data into training and prediction sets...")
        try:
            # Offset the features by the forecast_n_days_ahead for merging with labels
            features_df["offset_date"] = features_df["date"] + timedelta(
                days=self.forecast_n_days_ahead
            )

            # Merge on the offset dates
            data_df = features_df.merge(
                labels_df,
                left_on="offset_date",
                right_on="date",
                suffixes=("", "_label"),
            )
            data_df.drop(columns=["offset_date"], inplace=True)

            # Determine the last available date in features_df
            last_available_date = features_df["date"].max()

            # Training Data: Data up to the last available date minus forecast_n_days_ahead
            train_df = data_df[data_df["date"] <= last_available_date - timedelta(days=self.forecast_n_days_ahead)]

            # Calculate the start date for the prediction dataset
            prediction_start_date = last_available_date - timedelta(days=self.sequence_length - 1)

            predict_df = features_df[
                (features_df["date"] >= prediction_start_date) &
                (features_df["date"] <= last_available_date)
                ]

            # Check if predict_df has enough data points
            if predict_df.shape[0] < self.sequence_length:
                raise ValueError("Insufficient data for prediction. Required sequence length not met.")

        except Exception as e:
            logging.error(f"Failed to split data due to {e}.")
            raise
        else:
            logging.info("Data split completed.")
            return train_df, predict_df

    def _log_outdated_features(self, dfs: List[pd.DataFrame]) -> None:
        logging.info("Analyzing features...")
        try:
            latest_dates = {}
            for table, df in zip(self.table_names, dfs):
                if not df.empty:
                    latest_dates[table] = df['date'].max()
                else:
                    latest_dates[table] = None

            # Log the most outdated tables
            most_outdated = sorted(latest_dates.items(), key=lambda x: x[1] or pd.Timestamp.min)
            for table, date in most_outdated:
                logging.info(f"Table {table} last updated on {date}")

        except Exception as e:
            logging.error(f"Failed to analyze outdated features due to {e}.")
            raise

    def build_datasets(self):
        dataframes = self._get_data_from_tables()
        self._log_outdated_features(dataframes)
        aligned_data = self._align_dates(dataframes, "date")
        labels = self._get_labels()
        train, predict = self._split_data(aligned_data, labels)
        return train, predict


class ETFPredictor:
    def __init__(
            self,
            train_data,
            predict_data,
            sequence_length=28,
            epochs=1000,
            batch_size=32,
            stride=1,
            window_length=None,
            learning_rate=.001
    ):
        self.train_data = train_data.sort_values(by="date")
        self.predict_data = predict_data.sort_values(by="date")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.stride = stride
        self.window_length = window_length
        self.learning_rate = learning_rate

        self.scaler = StandardScaler()
        self.model = self._build_model()

        # Log the current arguments
        logging.info(
            f"Initialized ETFPredictor with sequence_length={sequence_length}, "
            f"epochs={epochs}, batch_size={batch_size}, stride={stride}"
        )

        logging.info(f"Training data is of size {train_data.shape} and prediction is of size {predict_data.shape}")

    def _build_model(self):
        model = Sequential()

        model.add(
            LSTM(
                units=self.train_data.shape[1],
                return_sequences=True,
                input_shape=(self.sequence_length, self.train_data.shape[1] - 2),
            )
        )
        model.add(LSTM(self.train_data.shape[1] // 2, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        logging.info(model.summary())
        return model

    def preprocess_data(self, split_ratio=0.9, validate=True):
        X = self.train_data.drop(columns=["date", "close"]).values if not self.window_length else self.train_data.drop(
            columns=["date", "close"]
        ).values[-self.window_length:]
        y = self.train_data["close"].values if not self.window_length else self.train_data["close"].values[
                                                                           -self.window_length:]

        X = self.scaler.fit_transform(X)

        # Scale labels
        y = y[self.sequence_length:]  # Align y with the sequences
        y = y.reshape(-1, 1)  # Reshape for scaling

        self.label_scaler = StandardScaler()  # Create a new scaler for labels
        y = self.label_scaler.fit_transform(y)

        sequences = [
            X[i: i + self.sequence_length]
            for i in range(0, len(X) - self.sequence_length)
        ]
        X = np.array(sequences)

        if validate:
            split_idx = int(split_ratio * len(X))
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            return X_train, y_train, X_val, y_val
        else:
            return X, y, None, None

    def train(self, validate=True):
        X_train, y_train, X_val, y_val = self.preprocess_data(validate=validate)
        early_stop = EarlyStopping(
            monitor="val_loss" if validate else "loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        )
        if validate:
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],

            )
        else:
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=1
            )

    def predict(self):
        # Prepare the last sequence for prediction
        last_sequence = self.predict_data.drop(columns=["date", "offset_date"]).values[-self.sequence_length:]
        last_sequence = self.scaler.transform(last_sequence)

        # Reshape the sequence to match the input shape expected by the model
        last_sequence = last_sequence.reshape((1, self.sequence_length, self.train_data.shape[1] - 2))

        # Use the model to predict the next value
        predicted_close = self.model.predict(last_sequence)[0][0]

        # If label scaling was used, inverse transform the prediction
        predicted_close = self.label_scaler.inverse_transform([[predicted_close]])[0][0]

        # Handling the date for the prediction
        dates_predict = self.predict_data["offset_date"].iloc[-1]
        future_date = dates_predict + timedelta(days=1)
        prediction_df = pd.DataFrame({"Date": [future_date], "Predicted_Close": [predicted_close]})

        return prediction_df

    def _rolling_train(self, train_window):
        """Trains the model on a rolling window from train_window."""
        # Extract features and target
        X = train_window.drop(columns=["date", "close"]).values
        y = train_window["close"].values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Create sequences for LSTM
        sequences = [
            X[i: i + self.sequence_length]
            for i in range(len(X) - self.sequence_length)
        ]
        X = np.array(sequences)

        # Scale labels
        y = y[self.sequence_length:]  # Align y with the sequences
        y = y.reshape(-1, 1)  # Reshape for scaling
        self.label_scaler = StandardScaler()  # Create a new scaler for labels
        y = self.label_scaler.fit_transform(y)

        early_stop = EarlyStopping(
            monitor="loss",
            patience=8,
            verbose=1,
            restore_best_weights=True,
        )

        # Fit the model
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, use_multiprocessing=False,
            callbacks=[early_stop]
        )

    def _sequence_predict(self, X):
        """Predicts the next value in the sequence."""
        # Ensure the input is correctly shaped
        if X.shape != (1, self.sequence_length, self.train_data.shape[1] - 2):
            raise ValueError(
                f"Expected X to have a shape of (1, {self.sequence_length}, {self.train_data.shape[1] - 2})."
            )

        # Get the prediction from the model
        prediction = self.model.predict(X)

        # Inverse transform the prediction to get it back on the original scale
        predicted_value = self.label_scaler.inverse_transform(prediction)

        return predicted_value[0][0]  # Return the scalar value

    def backtest(self, window_length=1000, overlap=500, days_ahead=7):
        if overlap >= window_length:
            raise ValueError("Overlap should be less than window length.")

        step_size = window_length - overlap
        n_windows = (len(self.train_data) - window_length) // step_size

        if n_windows <= 0:
            raise ValueError("Insufficient data for given window length and overlap.")

        results_df = pd.DataFrame(
            columns=[
                'Prediction_Date', 'Close_Price_On_Prediction_Date', 'Predicted_Close_Date',
                'Predicted_Close_Price', 'Actual_Close_Price', 'Predicted_Price_Change', 'Actual_Price_Change'
            ]
        )

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_length - self.sequence_length

            train_window = self.train_data.iloc[start_idx:end_idx]

            # Sequence for prediction is immediately after train_window
            prediction_sequence = self.train_data.iloc[end_idx:end_idx + self.sequence_length]

            prediction_date = prediction_sequence['date'].iloc[-1]

            # Finding the closing price on the prediction date (n days before in the dataset)
            close_price_on_prediction_date_index = self.train_data[
                self.train_data['date'] == prediction_date - timedelta(days=days_ahead)].index
            if close_price_on_prediction_date_index.empty:
                continue
            close_price_on_prediction_date = self.train_data.loc[close_price_on_prediction_date_index[0], 'close']

            # Actual close price is the known closing price n days after the prediction date
            actual_close_price = self.train_data.loc[
                self.train_data['date'] == prediction_date, 'close'
            ].values[0]

            self._rolling_train(train_window)

            # Prepare the sequence for prediction
            sequence = prediction_sequence.drop(columns=['date', 'close']).values
            X_next = self.scaler.transform(sequence).reshape(1, self.sequence_length, -1)
            predicted_close_price = self._sequence_predict(X_next)

            predicted_price_change = predicted_close_price - close_price_on_prediction_date
            actual_price_change = actual_close_price - close_price_on_prediction_date

            new_row = pd.DataFrame(
                {
                    'Prediction_Date': [prediction_date],
                    'Close_Price_On_Prediction_Date': [close_price_on_prediction_date],
                    'Predicted_Close_Date': [prediction_date + timedelta(days=days_ahead)],
                    'Predicted_Close_Price': [predicted_close_price],
                    'Actual_Close_Price': [actual_close_price],
                    'Predicted_Price_Change': [predicted_price_change],
                    'Actual_Price_Change': [actual_price_change]
                }
            )
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            print(results_df)
            self.model = self._build_model()

        mean_absolute_error = np.mean(np.abs(results_df['Actual_Close_Price'] - results_df['Predicted_Close_Price']))
        return mean_absolute_error, results_df




if __name__ == "__main__":
    import argparse
    import json


    def save_results(results, filename):
        # Extract directory from the filename
        directory = os.path.dirname(filename)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the file in the specified path
        with open(filename, 'w') as file:
            json.dump(results, file)


    # Create the parser
    parser = argparse.ArgumentParser(description="Build model for a given ETF with specified arguments")

    # Add arguments
    parser.add_argument("--etf", type=str, help="The ETF we are sourcing data for.")
    parser.add_argument("--days_ahead", type=int, default=182, help="How many days ahead are we forecasting?")
    parser.add_argument("--backtest", action="store_true", help="Are we back testing a model?")
    parser.add_argument("--sequence_length", type=int, default=28, help="Input sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training the model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model training.")
    parser.add_argument("--stride", type=int, default=14, help="Stride for training data preparation.")
    parser.add_argument(
        "--window_length", type=int, default=None, help="Length of the training window for backtesting."
    )
    parser.add_argument("--learning_rate", type=float, default=.001, help="Learning rate for model training")
    parser.add_argument(
        "--overlap", type=int, default=2500, help="Number of overlapping days in the training window for backtesting."
    )
    parser.add_argument("--validate", action="store_true", help="Enable validation during training.")

    # Parse the arguments
    args = parser.parse_args()

    # Extracting arguments
    etf_arg = args.etf
    days_ahead = args.days_ahead
    backtest = args.backtest
    sequence_length = args.sequence_length
    epochs = args.epochs
    batch_size = args.batch_size
    stride = args.stride
    window_length = args.window_length
    overlap = args.overlap

    file_path = f"./etf_feature_mappings/{etf_arg.lower()}.yml"
    config = load_config(filename=file_path)
    endpoints = config["endpoints"]

    tables = [endpoint.lower() for endpoint in endpoints.keys()]
    results = {
        'etf': etf_arg,
        'days_ahead': days_ahead,
        'backtest': backtest,
        'sequence_length': sequence_length,
        'epochs': epochs,
        'batch_size': batch_size,
        'stride': stride,
        'window_length': window_length,
        'overlap': overlap,
        'results': None
    }

    self = DatasetBuilder(
        table_names=tables, etf_symbol=etf_arg, forecast_n_days_ahead=days_ahead, sequence_length=sequence_length
    )

    train_data, predict_data = self.build_datasets()
    train_data.drop(columns="date_label", inplace=True)

    if args.backtest:
        self = ETFPredictor(
            train_data=train_data,
            predict_data=predict_data,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            stride=args.stride,
            learning_rate=args.learning_rate
        )

        # Backtest the predictor with overlapping windows:
        mae, results_df = self.backtest(
            window_length=args.window_length, overlap=args.overlap, days_ahead=args.days_ahead
        )
        print(results_df)
        results['results'] = {'mae': mae}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"./grid_search_results/{etf_arg}/{timestamp}.json")

        print(f"Mean Absolute Error during backtesting: {mae}")

    else:
        predictor = ETFPredictor(
            train_data=train_data,
            predict_data=predict_data,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            stride=args.stride,
            window_length=window_length,
            learning_rate=args.learning_rate

        )
        predictor.train(validate=args.validate)
        predictions = predictor.predict()
        print(predictions)
