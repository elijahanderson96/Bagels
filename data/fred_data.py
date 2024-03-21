import logging
from datetime import timedelta
from functools import reduce
from typing import List
from typing import Tuple

import pandas as pd

from database.database import db_connector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DatasetBuilder:
    def __init__(
        self,
        table_names: List[str],
        etf_symbol: str,
        forecast_n_days_ahead: int = 14,
        sequence_length: int = 14,
        from_date=None,
    ):
        self.table_names = table_names
        self.etf_symbol = etf_symbol
        self.forecast_n_days_ahead = forecast_n_days_ahead
        self.sequence_length = sequence_length
        self.from_date = from_date

    def _get_data_from_tables(self) -> List[pd.DataFrame]:
        logging.info("Fetching data from tables...")
        try:
            dfs = [
                db_connector.run_query(
                    f"SELECT * FROM {self.etf_symbol.lower()}.{table} WHERE date > '{self.from_date}'"
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
                f"symbol='{self.etf_symbol.upper()}'"
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

            # Create a DataFrame with date and close columns
            price_df = labels_df[["date", "close"]]

            # Calculate the highest and lowest closing prices within the forecast window
            data_df["etf_price_high"] = data_df.apply(
                lambda x: price_df[
                    (price_df["date"] > x["date"])
                    & (
                        price_df["date"]
                        <= x["date"] + timedelta(days=self.forecast_n_days_ahead)
                    )
                ]["close"].max(),
                axis=1,
            )
            data_df["etf_price_low"] = data_df.apply(
                lambda x: price_df[
                    (price_df["date"] > x["date"])
                    & (
                        price_df["date"]
                        <= x["date"] + timedelta(days=self.forecast_n_days_ahead)
                    )
                ]["close"].min(),
                axis=1,
            )

            # Determine the last available date in features_df
            last_available_date = features_df["date"].max()

            # Training Data: Data up to the last available date minus forecast_n_days_ahead
            train_df = data_df[
                data_df["date"]
                <= last_available_date - timedelta(days=self.forecast_n_days_ahead)
            ]

            # Calculate the start date for the prediction dataset
            prediction_start_date = last_available_date - timedelta(
                days=self.sequence_length - 1
            )

            predict_df = features_df[
                (features_df["date"] >= prediction_start_date)
                & (features_df["date"] <= last_available_date)
            ]

            # Check if predict_df has enough data points
            if predict_df.shape[0] < self.sequence_length:
                raise ValueError(
                    "Insufficient data for prediction. Required sequence length not met."
                )

        except Exception as e:
            logging.error(f"Failed to split data due to {e}.")
            raise
        else:
            logging.info("Data split completed.")
            train_df.drop(axis=1, inplace=True, labels=["date_label"])
            return train_df, predict_df

    def _log_outdated_features(self, dfs: List[pd.DataFrame]) -> None:
        logging.info("Analyzing features...")
        try:
            latest_dates = {}
            for table, df in zip(self.table_names, dfs):
                if not df.empty:
                    latest_dates[table] = df["date"].max()
                else:
                    latest_dates[table] = None

            # Log the most outdated tables
            most_outdated = sorted(
                latest_dates.items(), key=lambda x: x[1] or pd.Timestamp.min
            )
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
