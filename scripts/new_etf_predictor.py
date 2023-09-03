import logging
from datetime import datetime
from datetime import timedelta
from functools import reduce
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from database.database import db_connector
from scripts.ingestion_fred import endpoints


class DatasetBuilder:
    def __init__(
            self, table_names: List[str], etf_symbol: str, sequence_length: int = 14
    ):
        self.table_names = table_names
        self.etf_symbol = etf_symbol
        self.sequence_length = sequence_length

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
            self, features_df: pd.DataFrame, labels_df: pd.DataFrame
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
                days=self.sequence_length
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
                days=self.sequence_length
            )
            train_df = data_df[data_df["date"] <= last_training_date]
            predict_df = features_df[
                (features_df["date"] > last_training_date)
                & (
                        features_df["date"]
                        <= last_training_date + timedelta(days=self.sequence_length)
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


tables = [endpoint.lower() for endpoint in endpoints.keys()]
self = DatasetBuilder(table_names=tables, etf_symbol="SPY", sequence_length=7)
train_data, predict_data = self.build_datasets()
train_data.drop(columns="date_label", inplace=True)


class ETFPredictor:
    def __init__(
            self, train_data, predict_data, sequence_length=7, epochs=1000, batch_size=32
    ):
        self.train_data = train_data.sort_values(
            by="date"
        )  # Ensure data is sorted by date
        self.predict_data = predict_data.sort_values(by="date")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            LSTM(
                units=50, return_sequences=True, input_shape=(self.sequence_length, 44)
            )
        )
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def preprocess_data(self):
        # Scale features
        X = self.train_data.drop(columns=["date", "close"]).values
        y = self.train_data["close"].values

        X = self.scaler.fit_transform(X)

        # Reshape for LSTM
        X = np.array(
            [
                X[i: i + self.sequence_length]
                for i in range(len(X) - self.sequence_length)
            ]
        )
        y = y[self.sequence_length:]

        return X, y

    def train(self):
        X, y = self.preprocess_data()
        # Define early stopping callback
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stop])

    def predict(self):
        # Preprocess prediction data similarly to training data
        if len(self.predict_data) >= self.sequence_length:
            X_predict = self.predict_data.drop(columns=["date", "offset_date"]).values
            dates_predict = self.predict_data["offset_date"].values[
                            self.sequence_length - 1:
                            ]  # Extract dates for predictions

            # Ensure the date column is excluded before scaling
            X_predict = self.scaler.transform(X_predict)

            # Reshape X_predict to have the shape (batch_size, sequence_length, number_of_features)
            X_predict = np.reshape(
                X_predict, (1, self.sequence_length, -1)
            )  # -1 here will automatically use 44 (number of features)

            predictions = self.model.predict(X_predict)

            # Return the predictions alongside their corresponding dates
            return pd.DataFrame(
                {"Date": dates_predict, "Predicted_Close": predictions.flatten()}
            )
        else:
            print(
                "The prediction dataset doesn't have enough rows for the given sequence length."
            )
            return None


# Usage:
self = ETFPredictor(
    train_data=train_data,
    predict_data=predict_data,
    sequence_length=7,
    epochs=20000,
    batch_size=64,
)
self.train()
predictions = self.predict()

print(predictions)
