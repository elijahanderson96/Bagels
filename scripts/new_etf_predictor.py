import logging
from datetime import datetime
from datetime import timedelta
from functools import reduce
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from database.database import db_connector
from scripts.ingestion_fred import endpoints


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
                db_connector.run_query(f"SELECT * FROM data.{table}")
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
                f"SELECT date, close FROM data.historical_prices WHERE symbol='{self.etf_symbol}'"
            )
            labels_df.sort_values("date", inplace=True)
        except Exception as e:
            logging.error(f"Failed to fetch and process labels due to {e}.")
            raise
        else:
            logging.info("Label processing completed.")
            return labels_df

    def _split_data(
            self, features_df: pd.DataFrame, labels_df: pd.DataFrame, days_ahead: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info("Splitting data into training and prediction sets...")
        try:
            # Offset the features' date to present date only if the last_feature_date is behind today's date
            last_feature_date = features_df["date"].max()
            if last_feature_date < datetime.now():
                offset_days = (datetime.now() - last_feature_date).days
                features_df["date"] = features_df["date"] + timedelta(days=offset_days)

            # Offset the features by the sequence length for merging with labels
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
            self, train_data, predict_data, sequence_length=28, epochs=1000, batch_size=32, stride=1
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
        model.add(LSTM(units=25, return_sequences=True, input_shape=(self.sequence_length, 11)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(LSTM(units=5))
        model.add(Dense(units=self.sequence_length))
        model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_absolute_error"
        )
        return model

    def preprocess_data(self, split_ratio=.8, validate=True):
        X = self.train_data.drop(columns=["date", "close"]).values
        y = self.train_data["close"].values
        X = self.scaler.fit_transform(X)

        sequences = [X[i: i + self.sequence_length] for i in range(0, len(X) - self.sequence_length, self.stride)]
        X = np.array(sequences)
        y_sequences = [y[i: i + self.sequence_length] for i in range(0, len(y) - self.sequence_length, self.stride)]
        y = np.array(y_sequences)

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
            monitor='val_loss' if validate else 'loss', patience=25, verbose=1, restore_best_weights=True
            )
        if validate:
            self.model.fit(
                X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val),
                callbacks=[early_stop]
                )
        else:
            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stop])

    def predict(self, future_days=28):
        last_sequence = self.scaler.transform(self.predict_data.drop(columns=["date", "offset_date"]).values)[-self.sequence_length:]
        last_sequence = last_sequence.reshape((1, self.sequence_length, 11))
        predicted_sequence = self.model.predict(last_sequence)[0]
        dates_predict = self.predict_data["date"].iloc[-1]
        future_dates = pd.date_range(start=dates_predict, periods=future_days + 1)[1:]
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": predicted_sequence})

        return prediction_df


tables = [endpoint.lower() for endpoint in endpoints.keys()]
self = DatasetBuilder(table_names=tables, etf_symbol="AGG", forecast_n_days_ahead=28)
train_data, predict_data = self.build_datasets()
train_data.drop(columns="date_label", inplace=True)


# Usage:
self = ETFPredictor(
    train_data=train_data,
    predict_data=predict_data,
    sequence_length=28,
    epochs=20000,
    batch_size=1,
    stride=28
)
self.train(validate=True)
predictions = self.predict(future_days=28)

print(predictions)
