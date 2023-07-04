from datetime import datetime
from typing import Union

from database import PostgreSQLConnector


def fetch_actual_value(connector: PostgreSQLConnector, symbol: str, date: Union[str, datetime.date]) -> Union[
    float, None]:
    """
    Fetch the closing price from the 'historical_prices' table for a given symbol and date.

    Args:
        connector (PostgreSQLConnector): An instance of PostgreSQLConnector.
        symbol (str): The symbol.
        date (Union[str, datetime.date]): The date for which the closing price is to be fetched.

    Returns:
        Union[float, None]: The closing price for the given symbol and date, or None if not found.
    """
    query = """
    SELECT close
    FROM fred_raw.historical_prices
    WHERE symbol = %s AND date = %s
    """
    result = connector.run_query(query, params=(symbol, date), fetch_one=True)

    return result if result else None


def update_actual_values(connector: PostgreSQLConnector) -> None:
    """
    Update 'actual' values in the 'model_predictions' table.

    This function fetches the models where the date is in the past and the 'actual' value is not set,
    then updates the 'actual' value for these models in the 'model_predictions' table.

    Args:
        connector (PostgreSQLConnector): An instance of PostgreSQLConnector.
    """
    query = """
    SELECT mp.id, m.id as model_id, m.symbol, mp.date
    FROM forecasts.models m 
    LEFT JOIN forecasts.model_predictions MP
    ON mp.model_id = m.id
    WHERE mp.date < CURRENT_DATE
    AND mp.actual IS NULL
    """
    update_rows = connector.run_query(query)

    if update_rows.empty:
        return

    for _, row in update_rows.iterrows():
        try:
            actual_value = fetch_actual_value(connector, row['symbol'], row['date'])

            if actual_value is not None:
                # Update the 'actual' field in the 'model_predictions' table
                query = """
                UPDATE forecasts.model_predictions
                SET actual = %s
                WHERE id = %s AND model_id = %s AND date = %s
                """
                connector.run_query(
                    query, params=(actual_value, row['id'], row['model_id'], row['date']), return_df=False
                    )

                print(f"Updated {row['symbol']} with closing value of {actual_value} on date {row['date']}")

        except Exception as e:
            print(f"Failed to update actual value for model_id {row['id']} on date {row['date']}: {e}")


if __name__ == "__main__":
    from database import db_connector

    db_connector.connect()
    update_actual_values(db_connector)
