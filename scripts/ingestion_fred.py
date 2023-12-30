import datetime
import logging
import os
import time

import pandas as pd
import yaml
import yfinance
from fredapi import Fred

from database.database import db_connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    series = pd.DataFrame(fred.get_series_latest_release(endpoint))
    series = series.resample("D").asfreq()
    series = series.interpolate(method="linear")
    series = series.reset_index()
    series.rename(columns={0: "value", "index": "date"}, inplace=True)
    return series


def data_refresh(fred: Fred, etf: str, endpoints: dict) -> None:
    """Refreshes financial data.

    Function to refresh financial data by making API calls to different endpoints and then storing the results in a
    database.

    Args:
        fred (Fred): An instance of the Fred class.
        etf (str): The etf to get historical closing prices for.
        endpoints (dict): This is the config, as a dict, of the endpoints from fred we are using as inputs
        for predicting an etf's closing prices.

    Raises:
        Exception: If there is a failure in fetching the data from the API or inserting data into the database.

    Usage Example:
        >>> api_key = 'Your API Key'
        >>> data_refresh(api_key)
    """
    try:
        #    all_release_dates = fetch_all_release_dates()
        #    relevant_release_dates = [
        #        rd for rd in all_release_dates if rd["release_name"] in endpoints.values()
        #    ]
        #    next_release_dates_df = pd.DataFrame(relevant_release_dates)
        #
        #    db_connector.insert_dataframe(
        #        next_release_dates_df,
        #        schema=etf.lower(),
        #        name="next_release_dates",
        #        if_exists="replace",
        #        index=False,
        #    )

        metadata = pd.DataFrame(
            data={
                "id": range(0, len(endpoints)),
                "endpoint": endpoints.keys(),
                "value": endpoints.values(),
            }
        )
        db_kwargs = {
            "schema": etf.lower(),
            "name": "endpoints",
            "if_exists": "replace",
            "index": False,
        }

        db_connector.create_schema(etf.lower())

        db_connector.insert_dataframe(metadata, **db_kwargs)

        for endpoint in endpoints.keys():
            logger.info(f"Grabbing {endpoint}.")
            series = fetch_data(fred, endpoint)
            db_kwargs["name"] = endpoint.lower()
            db_connector.insert_dataframe(series, **db_kwargs)

        dfs = []

        logger.info(f"Grabbing {etf}.")
        time.sleep(1)
        df = yfinance.download(etf, end=datetime.datetime.now().date())
        df["symbol"] = etf
        df.reset_index(inplace=True)
        columns = {col: col.replace(" ", "_").lower() for col in df.columns.to_list()}
        df.rename(columns=columns, inplace=True)
        dfs.append(df)

        df = pd.concat(dfs)
        db_connector.insert_dataframe(
            df,
            schema=etf.lower(),
            name=f"{etf.lower()}_historical_prices",
            if_exists="replace",
            index=False,
        )

    except Exception as e:
        logger.error(f"Data refresh failed with error {e}")
        raise e


if __name__ == "__main__":
    import argparse

    api_key = os.getenv("FRED_API_KEY")

    parser = argparse.ArgumentParser(
        description="Fetch data based on ETF feature mapping configuration."
    )
    parser.add_argument("--etf", type=str, help="The ETF we are sourcing data for.")
    args = parser.parse_args()
    etf_arg = args.etf

    file_path = f"./etf_feature_mappings/{etf_arg.lower()}.yml"
    config = load_config(filename=file_path)

    endpoints = config["endpoints"]
    etf_arg = config["etf"][0]

    fred = Fred()
    data_refresh(fred, etf_arg, endpoints)
