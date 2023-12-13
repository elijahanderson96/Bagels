import logging
from datetime import timedelta
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from database.database import db_connector
from scripts.ingestion_fred import load_config


class DatasetBuilder:
    def __init__(
        self, table_names: List[str], etf_symbol: str, forecast_n_days_ahead: int = 14
    ):
        self.table_names = table_names
        self.etf_symbol = etf_symbol
        self.forecast_n_days_ahead = forecast_n_days_ahead

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

            # Split the data
            last_training_date = features_df["date"].max() - timedelta(
                days=self.forecast_n_days_ahead
            )
            train_df = data_df[data_df["date"] <= last_training_date]
            predict_df = features_df[
                (features_df["date"] > last_training_date)
                & (
                    features_df["date"]
                    <= last_training_date + timedelta(days=self.forecast_n_days_ahead)
                )
            ]

        except Exception as e:
            logging.error(f"Failed to split data due to {e}.")
            raise
        else:
            logging.info("Data split completed.")
            return train_df, predict_df

    def build_datasets(self):
        dataframes = self._get_data_from_tables()
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
    ):
        self.train_data = train_data.sort_values(by="date")
        self.predict_data = predict_data.sort_values(by="date")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.stride = stride

        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            LSTM(
                units=12,
                return_sequences=True,
                input_shape=(self.sequence_length, self.train_data.shape[1] - 2),
            )
        )
        model.add(Dropout(0.10))
        model.add(LSTM(units=10, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=0.002), loss="mean_absolute_error")
        return model

    def preprocess_data(self, split_ratio=0.9, validate=True):
        X = self.train_data.drop(columns=["date", "close"]).values
        y = self.train_data["close"].values
        X = self.scaler.fit_transform(X)

        sequences = [
            X[i : i + self.sequence_length]
            for i in range(0, len(X) - self.sequence_length)
        ]
        X = np.array(sequences)
        y = y[
            self.sequence_length :
        ]  # Now y contains the next value after each sequence

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
            patience=75,
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
            )

    def predict(self):
        last_sequence = self.scaler.transform(
            self.predict_data.drop(columns=["date"]).values
        )[-self.sequence_length :]
        last_sequence = last_sequence.reshape(
            (1, self.sequence_length, self.train_data.shape[1] - 2)
        )
        predicted_close = self.model.predict(last_sequence)[0][
            0
        ]  # Just one predicted value
        dates_predict = self.predict_data["date"].iloc[-1]
        future_date = dates_predict + timedelta(days=1)
        prediction_df = pd.DataFrame(
            {"Date": [future_date], "Predicted_Close": [predicted_close]}
        )

        return prediction_df

    def _rolling_train(self, train_window):
        """Trains the model on a rolling window from train_window."""
        X = train_window.drop(columns=["date", "close"]).values
        y = train_window["close"].values[1:]  # We're shifting y by one step

        X = self.scaler.fit_transform(X)
        sequences = [
            X[i : i + self.sequence_length]
            for i in range(len(X) - self.sequence_length - 1)
        ]
        X = np.array(sequences)

        y = y[self.sequence_length :]  # Match the shifted y size with X size
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            use_multiprocessing=True,
        )

    def _sequence_predict(self, X):
        # Ensure the input is reshaped to (1, sequence_length, number of features)
        if X.shape != (1, self.sequence_length, self.train_data.shape[1] - 2):
            raise ValueError(
                f"Expected X to have a shape of (1, {self.sequence_length}, {self.train_data.shape[1] - 2})."
            )

        prediction = self.model.predict(X)

        # Prepare the input for inverse transformation
        inverse_input = np.zeros((1, self.train_data.shape[1] - 2))
        inverse_input[:, :-1] = X[:, -1, :-1]
        inverse_input[:, -1] = prediction.ravel()

        # Denormalize the prediction
        predicted_value = self.scaler.inverse_transform(inverse_input)[:, -1]

        return predicted_value[0]

    def backtest(self, window_length=1000, overlap=500, days_ahead=7):
        """
        Conduct a backtest using a rolling window approach.

        Parameters:
        - window_length: Length of the training window.
        - overlap: Number of overlapping days in the training window.

        Returns:
        - mean_absolute_error: Mean absolute error across all rolling window predictions.
        """
        if overlap >= window_length:
            raise ValueError("Overlap should be less than window length.")

        step_size = (
            window_length - overlap
        )  # This determines the number of new days introduced in each window.
        n_windows = (len(self.train_data) - window_length) // step_size

        if n_windows <= 0:
            raise ValueError("Insufficient data for given window length and overlap.")

        errors = []

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_length

            train_window = self.train_data.iloc[start_idx:end_idx]
            # the next_data is the real value n_days ahead that we are trying to predict which will be used
            # to measure how good our model is
            next_data = self.train_data.iloc[
                end_idx + days_ahead : end_idx + days_ahead + 1
            ]
            print(train_window)
            print(next_data)
            if next_data.empty:
                continue

            self._rolling_train(train_window)
            print("Heres the prediction sequence below:")
            print(self.train_data.iloc[end_idx - self.sequence_length : end_idx])
            # Extract the feature matrix from next_data and predict the next value
            sequence = (
                self.train_data.iloc[end_idx - self.sequence_length : end_idx]
                .drop(columns=["date", "close"])
                .values
            )
            X_next = self.scaler.transform(sequence).reshape(
                1, self.sequence_length, -1
            )
            predicted_value = self._sequence_predict(X_next)

            # Measure the prediction error
            true_value = next_data["close"].values[0]
            error = np.abs(true_value - predicted_value)
            errors.append(error)
            print(f"predicted_value: {predicted_value}", f"true_value: {true_value}")
            self.model = self._build_model()  # Reset the model for the next iteration

        mean_absolute_error = np.mean(errors)
        return mean_absolute_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build model for a given ETF with specified arguments"
    )
    parser.add_argument("--etf", type=str, help="The ETF we are sourcing data for.")
    parser.add_argument(
        "--days_ahead",
        type=int,
        default=7,
        help="How many days ahead are we forecasting?",
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Are we back testing a model?"
    )

    args = parser.parse_args()
    etf_arg = args.etf
    days_ahead = args.days_ahead
    backtest = args.backtest

    mapping = {"SPY": "spy.yml", "QQQ": "qqq.yml", "AGG": "agg.yml"}
    file_path = f"./etf_feature_mappings/{mapping[etf_arg]}"
    config = load_config(filename=file_path)
    endpoints = config["endpoints"]

    tables = [endpoint.lower() for endpoint in endpoints.keys()]
    self = DatasetBuilder(
        table_names=tables, etf_symbol=etf_arg, forecast_n_days_ahead=days_ahead
    )

    train_data, predict_data = self.build_datasets()
    train_data.drop(columns="date_label", inplace=True)

    if args.backtest:
        predictor = ETFPredictor(
            train_data=train_data,
            predict_data=predict_data,
            sequence_length=14,  # this is the input sequence length, we predict the price n_days ahead.
            epochs=100,
            batch_size=4,
            stride=7,
        )

        # Backtest the predictor with overlapping windows:
        mae = predictor.backtest(window_length=1000, overlap=250, days_ahead=days_ahead)

        print(f"Mean Absolute Error during backtesting: {mae}")

    else:
        predictor = ETFPredictor(train_data=train_data, predict_data=predict_data)
        predictor.train(validate=True)
        predictions = predictor.predict()

        print(predictions)
        # Usage:
        self = ETFPredictor(
            train_data=train_data,
            predict_data=predict_data,
            sequence_length=7,
            epochs=2000,
            batch_size=4,
            stride=1,
        )
        self.train(validate=True)
        predictions = self.predict()

        print(predictions)
