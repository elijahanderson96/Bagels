import gzip
import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.src.optimizers import Adam
from keras.src.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler

from database.database import db_connector


class ETFPredictor:
    def __init__(
            self,
            etf,
            features,
            train_data,
            predict_data,
            sequence_length=28,
            epochs=1000,
            batch_size=32,
            stride=1,
            l1_kernel_regularizer=0.01,
            l2_kernel_regularizer=0.01,
            learning_rate=0.001,
            window_length=None,
            overlap=None,
            from_date=None,
            days_ahead=364,
    ):
        self.etf = etf.lower()
        self.features = features
        self.train_data = train_data.sort_values(by="date")
        self.predict_data = predict_data.sort_values(by="date")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.stride = stride
        self.window_length = window_length
        self.overlap = overlap
        self.learning_rate = learning_rate
        self.l1_kernel_regularizer = l1_kernel_regularizer
        self.l2_kernel_regularizer = l2_kernel_regularizer
        self.from_date = from_date
        self.days_ahead = days_ahead

        self.scaler = StandardScaler()
        self.label_scaler_close = StandardScaler()
        self.label_scaler_high = StandardScaler()
        self.label_scaler_low = StandardScaler()

        self.model = self._build_model()
        self.back_tested = False

        self.history = None
        self.n_windows = None

        # prediction df is the current prediction, backtest df is historical predictions
        self.prediction_df = None
        self.backtest_results_df = None

        # percent mean absolute error of the predicted close, high, and low prices.
        self.pmae = None
        self.pmae_high = None
        self.pmae_low = None

        # Log the current arguments
        logging.info(
            f"Initialized ETFPredictor with sequence_length={sequence_length}, "
            f"epochs={epochs}, batch_size={batch_size}, stride={stride}"
        )

        logging.info(
            f"Training data is of size {train_data.shape} and prediction is of size {predict_data.shape}"
        )

    def _build_model(self):
        clear_session()
        model = Sequential()

        model.add(
            LSTM(
                units=self.train_data.shape[1] - 4,
                return_sequences=True,
                input_shape=(self.sequence_length, self.train_data.shape[1] - 4),
                kernel_regularizer=l1_l2(
                    l1=self.l1_kernel_regularizer, l2=self.l2_kernel_regularizer
                ),
            )
        )
        model.add(
            LSTM(
                int(self.train_data.shape[1] // 1.5) + 4,
                kernel_regularizer=l1_l2(
                    l1=self.l1_kernel_regularizer, l2=self.l2_kernel_regularizer
                ),
                return_sequences=False,
            )
        )
        model.add(Dense(units=3))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        logging.info(model.summary())
        return model

    def preprocess_data(self):
        if self.window_length:
            train_data = self.train_data.iloc[-self.window_length:]
        else:
            train_data = self.train_data

        X = train_data.drop(
            columns=["date", "close", "etf_price_high", "etf_price_low"]
        ).values

        logging.info(f"New train data is of shape {len(X)}")

        y_columns = ["close", "etf_price_high", "etf_price_low"]
        y_data = {}

        label_scalers = {
            "close": self.label_scaler_close,
            "etf_price_high": self.label_scaler_high,
            "etf_price_low": self.label_scaler_low,
        }

        for column in y_columns:
            y_data[column] = train_data[column].values
            y_data[column] = y_data[column][self.sequence_length:]
            y_data[column] = y_data[column].reshape(-1, 1)
            y_data[column] = label_scalers[column].fit_transform(y_data[column])

        X = self.scaler.fit_transform(X)
        sequences = [
            X[i: i + self.sequence_length]
            for i in range(0, len(X) - self.sequence_length)
        ]
        X = np.array(sequences)
        y = np.column_stack(
            (y_data["close"], y_data["etf_price_high"], y_data["etf_price_low"])
        )
        return X, y

    def train(self, backtest=True):
        if backtest:
            self.backtest()

        self._build_model()

        X_train, y_train = self.preprocess_data()

        early_stop = EarlyStopping(
            monitor="loss",
            patience=20,
            verbose=0,
            restore_best_weights=True,
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

    def predict(self):
        # Prepare the last sequence for prediction
        last_sequence = self.predict_data.drop(columns=["date", "offset_date"]).values[
                        -self.sequence_length:
                        ]
        last_sequence = self.scaler.transform(last_sequence)

        # Reshape the sequence to match the input shape expected by the model
        last_sequence = last_sequence.reshape(
            (1, self.sequence_length, self.train_data.shape[1] - 4)
        )

        # Use the model to predict the next values
        predicted_values = self.model.predict(last_sequence)

        # Extract the predicted close, high, and low prices
        predicted_close = predicted_values[0][0]
        predicted_high = predicted_values[0][1]
        predicted_low = predicted_values[0][2]

        # If label scaling was used, inverse transform the predictions
        predicted_close = self.label_scaler_close.inverse_transform(
            [[predicted_close]]
        )[0][0]
        predicted_high = self.label_scaler_high.inverse_transform([[predicted_high]])[
            0
        ][0]
        predicted_low = self.label_scaler_low.inverse_transform([[predicted_low]])[0][0]

        # Handling the date for the prediction
        prediction_date = self.predict_data["date"].iloc[-1]
        dates_predict = self.predict_data["offset_date"].iloc[-1]
        future_date = dates_predict + timedelta(days=1)

        self.prediction_df = pd.DataFrame(
            {
                "Date": [future_date],
                "Predicted_Close": [predicted_close],
                "Predicted_High": [predicted_high],
                "Predicted_Low": [predicted_low],
                "Prediction_Made_On_Date": prediction_date,
            }
        )

        return self.prediction_df

    def save_experiment(self):
        if not self.backtest:
            return

        model_id = self.save_model_details()
        pass

    def save_model_details(self):
        training_loss = json.dumps(
            {
                f"Epoch {i + 1}": loss
                for i, loss in enumerate(self.history.history["loss"])
            }
        )

        model_details = {
            "trained_on_date": pd.Timestamp("now").strftime("%Y-%m-%d"),
            "features": json.dumps(self.features),
            "architecture": json.dumps(self.model.get_config()),
            "hyperparameters": json.dumps(
                {
                    "window_length": self.window_length,
                    "overlap": self.overlap,
                    "sequence_length": self.sequence_length,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "stride": self.stride,
                    "learning_rate": self.learning_rate,
                    "from_date": self.from_date,
                    "l1_regularization": self.l1_kernel_regularizer,
                    "l2_regularization": self.l2_kernel_regularizer,
                }
            ),
            "training_loss_info": training_loss,
        }

        model_id = db_connector.insert_and_return_id(
            "models", model_details, schema=self.etf
        )

        return model_id

    def save_model_predictions(self, schema, model_id, prediction_dataframe):
        """
        Save model predictions to the 'model_predictions' table.

        Args:
            schema (str): The schema name where the 'model_predictions' table exists.
            model_id (int): The ID of the model.
            prediction_dataframe (pd.DataFrame): DataFrame containing prediction data.
        """
        for index, row in prediction_dataframe.round(2).iterrows():
            prediction_details = {
                "model_id": model_id,
                "date": row["Date"].strftime("%Y-%m-%d"),
                "predicted_price": row["Predicted_Close"],
                "prediction_made_on_date": row["Prediction_Made_On_Date"].strftime(
                    "%Y-%m-%d"
                ),
            }
            db_connector.insert_row("forecasts", prediction_details, schema=schema)

    def save_backtest_results(
            self,
            schema,
            model_id,
            mape,
            cap,
            training_windows,
            mpae_range,
            results_df,
    ):
        """
        Save backtest results to the 'backtest_results' table.

        Args:
            schema (str): The schema name where the 'backtest_results' table exists.
            model_id (int): The ID of the model.
            mape (float): Mean Absolute Percentage Error.
            cap (float): Classification Accuracy Percentage.
            training_windows (int): Number of training windows.
            mpae_range (str): MPAE price range.
            results_df (DataFrame): Compressed DataFrame containing backtest data.
        """
        backtest_details = {
            "model_id": model_id,
            "mean_absolute_percentage_error": mape,
            "classification_accuracy_percentage": cap,
            "number_of_training_windows": training_windows,
            "mpae_price_range": mpae_range,
            "data_blob": gzip.compress(
                results_df.round(2).to_csv(index=False).encode()
            ),
        }

        db_connector.insert_row("backtest_results", backtest_details, schema=schema)

    def save_data(self, model_id, df, table_name, schema):
        compressed_data = gzip.compress(df.round(2).to_csv(index=False).encode())
        data = {"model_id": model_id, "data_blob": compressed_data}
        db_connector.insert_row(table_name, data, schema=schema)

    def _rolling_train(self, train_window):
        """Trains the model on a rolling window from train_window."""
        # Extract features and targets
        X = train_window.drop(
            columns=["date", "close", "etf_price_high", "etf_price_low"]
        ).values
        y_close = train_window["close"].values
        y_high = train_window["etf_price_high"].values
        y_low = train_window["etf_price_low"].values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Create sequences for LSTM
        sequences = [
            X[i: i + self.sequence_length]
            for i in range(len(X) - self.sequence_length)
        ]
        X = np.array(sequences)

        # Align targets with the sequences
        y_close = y_close[self.sequence_length:]
        y_high = y_high[self.sequence_length:]
        y_low = y_low[self.sequence_length:]

        # Reshape targets for scaling
        y_close = y_close.reshape(-1, 1)
        y_high = y_high.reshape(-1, 1)
        y_low = y_low.reshape(-1, 1)

        # Create new scalers for labels
        self.label_scaler_close = StandardScaler()
        self.label_scaler_high = StandardScaler()
        self.label_scaler_low = StandardScaler()

        # Scale labels
        y_close = self.label_scaler_close.fit_transform(y_close)
        y_high = self.label_scaler_high.fit_transform(y_high)
        y_low = self.label_scaler_low.fit_transform(y_low)

        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            verbose=0,
            restore_best_weights=True,
        )

        # Fit the model
        self.model.fit(
            X,
            [y_close, y_high, y_low],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            use_multiprocessing=False,
            callbacks=[early_stop],
        )

    def _sequence_predict(self, X):
        """Predicts the next values in the sequence."""
        # Ensure the input is correctly shaped
        if X.shape != (1, self.sequence_length, self.train_data.shape[1] - 4):
            raise ValueError(
                f"Expected X to have a shape of (1, {self.sequence_length}, {self.train_data.shape[1] - 4})."
            )

        # Get the predictions from the model
        predictions = self.model.predict(X)

        # Inverse transform the predictions to get them back on the original scale
        predicted_close = self.label_scaler_close.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()[0]
        predicted_high = self.label_scaler_high.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()[0]
        predicted_low = self.label_scaler_low.inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten()[0]

        return predicted_close, predicted_high, predicted_low

    def backtest(self):
        self.back_tested = True

        if self.overlap >= self.window_length:
            raise ValueError("Overlap should be less than window length.")

        step_size = self.window_length - self.overlap
        self.n_windows = (len(self.train_data) - self.window_length) // step_size

        logging.info(
            f"The window length is {self.window_length}, overlap is {self.overlap}. This means we have a step size of "
            f"{step_size} and there are {self.n_windows} windows we will be iterating over."
        )

        if self.n_windows <= 0:
            raise ValueError("Insufficient data for given window length and overlap.")

        results = []

        for i in range(self.n_windows):
            logging.info(
                f"We are currently on window {i + 1} of {self.n_windows + 1} training windows"
            )

            start_idx = i * step_size
            end_idx = start_idx + self.window_length - self.sequence_length

            train_window = self.train_data.iloc[start_idx:end_idx]

            # Sequence for prediction is immediately after train_window
            prediction_sequence = self.train_data.iloc[
                                  end_idx: end_idx + self.sequence_length
                                  ]

            prediction_date = prediction_sequence["date"].iloc[-1]
            predicted_close_date = prediction_date + timedelta(days=self.days_ahead)

            # Finding the closing price on the prediction date (n days before in the dataset)
            close_price_on_prediction_date = self.train_data.loc[
                self.train_data["date"] == prediction_date - timedelta(days=self.days_ahead), "close",
            ].values

            if not close_price_on_prediction_date:
                continue

            close_price_on_prediction_date = close_price_on_prediction_date[0]

            # Actual prices are the known prices n days after the prediction date
            actual_prices = self.train_data.loc[self.train_data["date"] == prediction_date]

            actual_close_price = actual_prices["close"].values[0]
            actual_high_price = actual_prices["etf_price_high"].values[0]
            actual_low_price = actual_prices["etf_price_low"].values[0]

            self._rolling_train(train_window)

            # Prepare the sequence for prediction
            sequence = prediction_sequence[[_.lower() for _ in self.features.keys()]].values
            X_next = self.scaler.transform(sequence).reshape(1, self.sequence_length, -1)

            predicted_close_price, predicted_high_price, predicted_low_price = self._sequence_predict(X_next)

            predicted_price_change = predicted_close_price - close_price_on_prediction_date
            actual_price_change = actual_close_price - close_price_on_prediction_date

            results.append(
                {
                    "prediction_date": prediction_date,
                    "close_price_on_prediction_date": close_price_on_prediction_date,
                    "predicted_close_date": predicted_close_date,
                    "predicted_close_price": predicted_close_price,
                    "predicted_high_price": predicted_high_price,
                    "predicted_low_price": predicted_low_price,
                    "actual_close_price": actual_close_price,
                    "actual_high_price": actual_high_price,
                    "actual_low_price": actual_low_price,
                    "predicted_price_change": predicted_price_change,
                    "actual_price_change": actual_price_change,
                }
            )

            self.model = self._build_model()

        self.backtest_results_df = pd.DataFrame(results)
        self.backtest_results_df, self.pmae, self.pmae_high, self.pmae_low = \
            self.calculate_percent_mean_absolute_error()

        logging.info(f"The mean absolute error percentage is: {self.pmae}%")
        logging.info(f"The mean absolute error percentage for the high: {self.pmae_high}%")
        logging.info(
            f"The mean absolute error percentage for the low: {self.pmae_low}%"
        )

        return self.backtest_results_df, self.pmae, self.pmae_high, self.pmae_low

    def calculate_percent_mean_absolute_error(self):
        price_types = ["close", "high", "low"]
        percentage_errors = {}
        pmae_values = {}

        for price_type in price_types:
            predicted_col = f"predicted_{price_type}_price"
            actual_col = f"actual_{price_type}_price"

            differences = (
                    self.backtest_results_df[predicted_col]
                    - self.backtest_results_df[actual_col]
            )
            percentage_errors[price_type] = (
                    differences / self.backtest_results_df[actual_col]
            )

            pmae_values[price_type] = (
                    np.mean(np.abs(percentage_errors[price_type])) * 100
            )

        self.pmae = pmae_values["close"]
        self.pmae_high = pmae_values["high"]
        self.pmae_low = pmae_values["low"]

        return self.backtest_results_df, self.pmae, self.pmae_high, self.pmae_low

    def adjusted_prediction_range(self):
        if self.prediction_df is None:
            raise ValueError(
                "No prediction data available. Make sure to call predict() method first."
            )

        if self.pmae is None or self.pmae_high is None or self.pmae_low is None:
            raise ValueError(
                "PMAE values are not available. Make sure to calculate PMAE values first."
            )

        prediction_data = {
            "Date": self.prediction_df["Date"],
            "Prediction_Made_On_Date": self.prediction_df["Prediction_Made_On_Date"],
        }

        for price_type in ["Close", "High", "Low"]:
            predicted_price = self.prediction_df[f"Predicted_{price_type}"].iloc[0]
            pmae = getattr(self, f"pmae_{price_type.lower()}")

            mae_adjustment = pmae / 100 * predicted_price
            lower_bound = predicted_price - mae_adjustment
            upper_bound = predicted_price + mae_adjustment

            prediction_data[f"Predicted_{price_type}"] = predicted_price
            prediction_data[f"Lower_Bound_{price_type}"] = round(lower_bound, 2)
            prediction_data[f"Upper_Bound_{price_type}"] = round(upper_bound, 2)

        adjusted_prediction_df = pd.DataFrame(prediction_data)

        return adjusted_prediction_df

    def evaluate_directional_accuracy(self, backtest_results_df: pd.DataFrame) -> float:
        """
        Evaluate the accuracy of the model in predicting the direction of price changes.

        Args:
            backtest_results_df (pd.DataFrame): DataFrame containing the backtest results.
                                                It must have 'predicted_price_change' and 'actual_price_change' columns.

        Returns:
            float: The accuracy of the model in predicting the direction of the price change.

        Raises:
            ValueError: If the input DataFrame is empty or not properly formatted.
        """
        if backtest_results_df.empty:
            raise ValueError("Backtest results DataFrame is empty.")

        if not {"predicted_price_change", "actual_price_change"}.issubset(
                backtest_results_df.columns
        ):
            raise ValueError(
                "DataFrame must contain 'predicted_price_change' and 'actual_price_change' columns."
            )

        correct_predictions = (
                backtest_results_df["predicted_price_change"]
                * backtest_results_df["actual_price_change"]
                > 0
        )

        accuracy = correct_predictions.mean() * 100
        logging.info(
            f"The accuracy of the model when classifying price increase or decrease is {accuracy} %"
        )
        return accuracy
