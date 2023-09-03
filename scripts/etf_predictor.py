import logging
from datetime import datetime
from datetime import timedelta
from functools import reduce
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from database.database import db_connector
from scripts.ingestion_fred import endpoints
from scripts.ingestion_fred import etfs

# Setting up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ETFPredictor:
    def __init__(
        self,
        table_names: List[str],
        etf_symbol: str,
        days_forecast: int,
        input_seq_length: int = 14,
        output_seq_length: int = 14,
    ):
        self.n_training_points = None
        self.trained_to = None
        self.trained_from = None
        self.table_names = table_names
        self.etf_symbol = etf_symbol
        self.days_forecast = days_forecast
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.model = None

    def _generate_sequences(self, data, labels):
        """
        Generate sequences for LSTM input and output
        """
        X, y = [], []
        for i in range(len(data) - self.input_seq_length - self.output_seq_length + 1):
            X.append(data[i : i + self.input_seq_length])
            y.append(
                labels[
                    i
                    + self.input_seq_length : i
                    + self.input_seq_length
                    + self.output_seq_length
                ]
            )
        return np.array(X), np.array(y)

    def _align_dates(
        self, df_list: List[pd.DataFrame], date_column: str
    ) -> pd.DataFrame:
        """
        Align dates across different dataframes in the list

        Args:
            df_list (List[pd.DataFrame]): List of dataframes
            date_column (str): Name of the column containing dates

        Returns:
            pd.DataFrame: Merged DataFrame with aligned dates
        """
        logging.info("Aligning dates across dataframes...")
        try:
            # Convert date columns to datetime and set as index
            for i, df in enumerate(df_list):
                df[date_column] = df[date_column].astype("datetime64[ns]")
                df.set_index(date_column, inplace=True)
                df_list[i] = df

            # Merge all dataframes in the list based on the date index
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

    def _get_labels(self) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Fetch labels from the database and generate binary labels

        Returns:
            Tuple[pd.DataFrame, MinMaxScaler]: DataFrame with binary labels and closing price at prediction
                                                and fitted scaler for labels
        """
        logging.info("Fetching and processing labels...")
        try:
            # TODO bring out this query to the separate file.
            labels_df = db_connector.run_query(
                "SELECT date, close "
                "FROM data.historical_prices "
                f"WHERE symbol='{self.etf_symbol}'"
            )

            labels_df.sort_values("date", inplace=True)

            # Scale the labels
            scaler = MinMaxScaler()
            labels_df["close"] = scaler.fit_transform(
                labels_df["close"].values.reshape(-1, 1)
            )

        except Exception as e:
            logging.error(f"Failed to fetch and process labels due to {e}.")
            raise
        else:
            logging.info("Label processing completed.")
            return labels_df, scaler

    def _split_data(
        self, features_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and prediction sets

        Args:
            features_df (pd.DataFrame): DataFrame with features
            labels_df (pd.DataFrame): DataFrame with labels

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_data, test_data, and closing_price_at_prediction
            DataFrames
        """
        logging.info("Splitting data into training and prediction sets...")

        try:
            features_df["date"] = features_df["date"] + timedelta(
                days=self.days_forecast
            )

            data_df = features_df.merge(labels_df, how="left", on="date")
            data_df.dropna(subset=["close"], inplace=True)

            today = datetime.now()

            train_df = data_df[data_df["date"] <= today]
            predict_df = features_df[features_df["date"] > today]

            predict_df.drop_duplicates(subset=["date"], inplace=True)
            predict_df.drop_duplicates(subset=["date"], inplace=True)

        except Exception as e:
            logging.error(f"Failed to split data due to {e}.")
            raise
        else:
            logging.info("Data split completed.")
            return train_df, predict_df

    def _preprocess_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, MinMaxScaler]:
        """
        Preprocesses the data for LSTM

        Args:
            df (pd.DataFrame): DataFrame to preprocess

        Returns:
            Tuple[np.ndarray, pd.Series, MinMaxScaler]: Processed data, labels, fitted scaler
        """
        logging.info("Preprocessing data...")
        try:
            df = df.set_index("date")
            df.sort_index()
            df.dropna(inplace=True)

            labels = df.pop("close")

            # Scale the features to be between 0 and 1
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

            # Convert the dataframe into sequences for LSTM
            data, labels = self._generate_sequences(df.values, labels.values)

        except Exception as e:
            logging.error(f"Failed to preprocess data due to {e}.")
            raise

        else:
            logging.info("Data preprocessing completed.")
            return data, labels, scaler

    def _define_and_train_model(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        validate=False,
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
        logging.info("Defining the LSTM model...")
        self.model = Sequential()

        self.model.add(
            LSTM(128, return_sequences=True, input_shape=(self.input_seq_length, 44))
        )
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dense(1, activation="linear"))

        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001), loss="mean_squared_error"
        )

        # Define early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss" if validate else "loss",
            patience=8,
            mode="min",
            restore_best_weights=True,
        )

        # Fit the model
        logging.info("Training the model...")
        try:
            history = self.model.fit(
                X_train,
                y_train,
                epochs=5000,
                validation_data=(X_test, y_test) if validate else None,
                verbose=1,
                callbacks=[early_stopping],
            )

        except Exception as e:
            logging.error(f"Failed to train the model due to {e}.")
            raise
        else:
            logging.info("Model training completed.")
            self.model_metadata = self.model.summary()
            print(self.model_metadata)
            return history

    def _predict(
        self,
        predict_data: pd.DataFrame,
        scaler: MinMaxScaler,
        label_scaler: MinMaxScaler,
        input_sequence_length: int = 14,  # default values
        output_sequence_length: int = 14,  # default values
    ) -> np.ndarray:
        """
        Generate predictions with the trained model

        Args:
            predict_data (pd.DataFrame): Data to predict
            scaler (MinMaxScaler): Fitted scaler for feature renormalization
            label_scaler (MinMaxScaler): Fitted scaler for label renormalization
            input_sequence_length (int): Length of input sequences
            output_sequence_length (int): Length of output sequences

        Returns:
            np.ndarray: Predictions
        """
        logging.info("Making predictions...")
        try:
            # Set date as index and sort
            predict_data = predict_data.set_index("date")
            predict_data.sort_index()
            predict_data.dropna(inplace=True)
            # Scale the features
            predict_data_scaled = scaler.transform(predict_data)
            # Create batches for predictions
            total_sequences = len(predict_data) - input_sequence_length + 1
            input_batches = np.zeros(
                (total_sequences, input_sequence_length, predict_data.shape[1])
            )
            for i in range(total_sequences):
                input_batches[i] = predict_data_scaled[i : i + input_sequence_length]
            # Predict
            y_pred = self.model.predict(input_batches)
            # We might be predicting multiple values (output_sequence_length) for a single input sequence
            y_pred_reshaped = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
            y_pred_inverse_scaled = label_scaler.inverse_transform(y_pred_reshaped)
            return y_pred_inverse_scaled
        except Exception as e:
            logging.error(f"Failed to make predictions due to {e}.")
            raise

    def _store_predictions(self, predictions_df: pd.DataFrame, model_id: int) -> None:
        """
        Store model predictions in the database.

        Args:
            predictions_df (pd.DataFrame): DataFrame with model predictions
            model_id (int): The associated model id.
        """
        logging.info("Storing model predictions...")
        try:
            predictions_df["model_id"] = model_id
            predictions_df["actual"] = None
            kwargs = {
                "name": "model_predictions",
                "if_exists": "append",
                "schema": "models",
                "index": False,
            }
            db_connector.insert_dataframe(predictions_df, **kwargs)
        except Exception as e:
            logging.error(f"Failed to store model predictions due to {e}.")
            raise
        else:
            logging.info("Model predictions stored successfully.")

    def _store_model(self, history) -> int:
        """
        Store model metadata in the database
        """
        logging.info("Storing model metadata...")
        try:
            model = pd.DataFrame(
                {
                    "trained_from": [self.trained_from],
                    "trained_to": [self.trained_to],
                    "n_training_points": [self.n_training_points],
                    "date_trained": [datetime.now().date()],
                    "symbol": [self.etf_symbol],
                    # early stopping should return the lowest loss value since we're keeping best weights.
                    "loss": [min(history.history["loss"])],
                    "features": [", ".join(self.table_names)],
                    "days_forecast": [self.days_forecast],
                    "model_summary": [self.model_metadata],
                }
            )
            model_id = db_connector.insert_and_return_id(
                table_name="models",
                columns=model.to_dict("records")[0],
                schema="models",
            )
        except Exception as e:
            logging.error(f"Failed to store model metrics due to {e}.")
            raise
        else:
            logging.info("Model metrics stored successfully.")
            return model_id

    def predict(self, validate=False):
        dfs = self._get_data_from_tables()
        features_df = self._align_dates(dfs, "date")
        labels_df, labels_scaler = self._get_labels()
        train_df, predict_df = self._split_data(features_df, labels_df)

        self.trained_from = min(train_df["date"])
        self.trained_to = max(train_df["date"])
        self.n_training_points = len(train_df)

        train_data, train_labels, feature_scaler = self._preprocess_data(train_df)

        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_labels, test_size=0.2
            )

        else:
            X_train, y_train = train_data, train_labels
            X_val, y_val = None, None

        history = self._define_and_train_model(X_train, y_train, X_val, y_val)
        predictions = self._predict(predict_df, feature_scaler, labels_scaler)

        result_df = pd.DataFrame(
            {"date": predict_df["date"].to_list(), "prediction": predictions.flatten()}
        )

        result_df.drop_duplicates(subset=["date"], inplace=True)

        model_id = self._store_model(history)
        self._store_predictions(result_df, model_id)


if __name__ == "__main__":
    tables = [endpoint.lower() for endpoint in endpoints.keys()]
    # data_refresh(os.getenv("FRED_API_KEY"))
    for etf in etfs:
        predictor = ETFPredictor(
            table_names=tables, etf_symbol=etf, days_forecast=7 * 52
        )
        predictor.predict()
