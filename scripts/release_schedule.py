import os

import pandas as pd
import requests

from database.database import db_connector
from scripts.ingestion_fred import load_config

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"


def get_series_code_for_release(release_id):
    endpoint = f"{FRED_API_BASE_URL}/release/series"
    params = {
        "api_key": os.getenv("FRED_API_KEY"),
        "release_id": release_id,
        "file_type": "json",
    }
    response = requests.get(endpoint, params=params)
    data = response.json()

    return pd.DataFrame(data["seriess"])


def fetch_release_id_for_series(series_code):
    endpoint = f"{FRED_API_BASE_URL}/series/release"
    params = {
        "api_key": os.getenv("FRED_API_KEY"),
        "series_id": series_code,
        "file_type": "json",
    }
    response = requests.get(endpoint, params=params)
    data = response.json()

    if "releases" in data and len(data["releases"]) > 0:
        return data["releases"][0].get("id")
    return None


def fetch_all_release_dates(release_ids=None):
    endpoint = f"{FRED_API_BASE_URL}/releases/dates"
    params = {"api_key": os.getenv("FRED_API_KEY"), "file_type": "json"}
    response = requests.get(endpoint, params=params)
    data = response.json()

    if release_ids:
        data["release_dates"] = [
            item for item in data["release_dates"] if item["release_id"] in release_ids
        ]

    return data["release_dates"]


def get_most_recent_dates(release_dates_data):
    """
    Get the most recent release date for each release_id.

    Args:
        release_dates_data (dict): The original release_dates_data containing all dates.

    Returns:
        df: A dictionary with release_id as keys and their most recent release_date as values.
    """
    dates_df = pd.DataFrame(release_dates_data)
    return dates_df.groupby("release_id")["date"].max().reset_index()


def update_release_schedule_for_schema(schema_name: str, release_data_df: pd.DataFrame):
    check_and_create_release_schedule(
        schema_name
    )  # Check and create release_schedule table if not exists

    for _, row in release_data_df.iterrows():
        release_id, release_date, endpoint = row

        query = f"SELECT release_id, release_date FROM {schema_name}.release_schedule WHERE endpoint_name = %s"
        current_data = db_connector.run_query(query, (endpoint,))
        print(current_data)
        if not current_data.empty:
            current_release_id, current_release_date = (
                current_data.iloc[0, 0],
                current_data.iloc[0, 1],
            )
            if release_date > current_release_date or release_id != current_release_id:
                update_query = f"""
                UPDATE {schema_name}.release_schedule 
                SET release_id = %s, release_date = %s 
                WHERE endpoint_name = %s
                """
                db_connector.run_query(
                    update_query, (release_id, release_date, endpoint), return_df=False
                )
        else:
            insert_query = f"""
            INSERT INTO {schema_name}.release_schedule (endpoint_name, release_id, release_date) 
            VALUES (%s, %s, %s)
            """
            db_connector.run_query(
                insert_query, (endpoint, release_id, release_date), return_df=False
            )


def check_and_create_release_schedule(schema_name: str):
    """
    Check if release_schedule table exists in the given schema. If not, create it.
    """
    check_table_query = f"""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = '{schema_name}' 
        AND table_name = 'release_schedule'
    );
    """

    table_exists = db_connector.run_query(check_table_query, fetch_one=True)

    if not table_exists:
        create_table_query = f"""
        CREATE TABLE {schema_name}.release_schedule (
            endpoint_name VARCHAR(255) PRIMARY KEY,
            release_id INT NOT NULL,
            release_date DATE
        );
        """

        db_connector.run_query(create_table_query, return_df=False)


def main():
    mapping = {"SPY": "spy.yml", "AGG": "agg.yml"}
    for etf, yml in mapping.items():
        config = load_config(filename=f"./etf_feature_mappings/{yml}")
        endpoints = config["endpoints"]

        # release_ids contain many endpoints per release (release_id -> endpoints is one to many)
        # dict of form ENDPOINT_NAME: RELEASE_ID
        endpoint_release_ids = {
            endpoint: fetch_release_id_for_series(endpoint)
            for endpoint in endpoints.keys()
        }
        endpoint_release_ids_dataframe = pd.DataFrame(
            [
                {"endpoint_name": key, "release_id": value}
                for key, value in endpoint_release_ids.items()
            ]
        )

        release_ids = list(set(endpoint_release_ids.values()))
        release_dates_data = fetch_all_release_dates(release_ids)

        recent_dates = get_most_recent_dates(release_dates_data)

        endpoint_name_release_id_date_mapping = recent_dates.merge(
            endpoint_release_ids_dataframe, on="release_id"
        )

        update_release_schedule_for_schema(
            etf.lower(), endpoint_name_release_id_date_mapping
        )


if __name__ == "__main__":
    main()
