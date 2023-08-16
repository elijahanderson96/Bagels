import datetime
import logging
import time

import pandas as pd
import yaml
import yfinance
from fredapi import Fred

from database import db_connector


def load_config(filename: str) -> dict:
    """
    Load the configuration data from a YAML file.

    Args:
        filename (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the configuration data.
    """
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config("pipeline_metadata.yml")
endpoints = config["endpoints"]
etfs = config["etfs"]


def fetch_data(fred: Fred, endpoint: str) -> pd.DataFrame:
    """Fetches data for a given endpoint from the FRED API.

    Args:
        fred (Fred): An instance of the Fred class.
        endpoint (str): The FRED API endpoint to fetch data from.

    Returns:
        pd.DataFrame: A DataFrame with the fetched data.

    Usage Example:
        >>> fred = Fred(api_key='Your API Key')
        >>> endpoint = 'DGS1'
        >>> fetch_data(fred, endpoint)
    """
    # Sleep so we don't exceed API Usage calls.
    time.sleep(0.5)
    series = pd.DataFrame(fred.get_series(endpoint))
    series = series.resample("D").asfreq()
    series = series.interpolate(method="linear")

    series = series.reset_index()
    series.rename(columns={0: "value", "index": "date"}, inplace=True)

    return series


def data_refresh(api_key: str) -> None:
    """Refreshes financial data.

    Function to refresh financial data by making API calls to different endpoints and then storing the results in a
    database.

    Args:
        api_key (str): FRED API key.

    Raises:
        Exception: If there is a failure in fetching the data from the API or inserting data into the database.

    Usage Example:
        >>> api_key = 'Your API Key'
        >>> data_refresh(api_key)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        fred = Fred(api_key=api_key)
        metadata = pd.DataFrame(
            data={
                "id": range(0, len(endpoints)),
                "endpoint": endpoints.keys(),
                "value": endpoints.values(),
            }
        )
        db_kwargs = {
            "schema": "fred_raw",
            "name": "endpoints",
            "if_exists": "replace",
            "index": False,
        }
        db_connector.insert_dataframe(metadata, **db_kwargs)

        for endpoint in endpoints.keys():
            logger.info(f"Grabbing {endpoint}.")
            series = fetch_data(fred, endpoint)
            db_kwargs["name"] = endpoint.lower()
            db_connector.insert_dataframe(series, **db_kwargs)

        dfs = []

        for etf in etfs:
            logger.info(f"Grabbing {etf}.")
            time.sleep(1)
            df = yfinance.download(etf, end=datetime.datetime.now().date())
            df["symbol"] = etf
            df.reset_index(inplace=True)
            [
                df.rename(columns={col: col.replace(" ", "_").lower()}, inplace=True)
                for col in df.columns.to_list()
            ]
            dfs.append(df)

        df = pd.concat(dfs)
        db_connector.insert_dataframe(
            df,
            schema="fred_raw",
            name="historical_prices",
            if_exists="replace",
            index=False,
        )

    except Exception as e:
        logger.error(f"Data refresh failed with error {e}")
        raise e


if __name__ == "__main__":
    # Your FRED API Key
    api_key = "7f54d62f0a53c2b106b903fc80ecace1"
    data_refresh(api_key)
