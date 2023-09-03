# set up the database from scratch.
from database import PostgreSQLConnector

connector = PostgreSQLConnector(
    host="localhost",
    port="5432",
    user="postgres",
    password="password",
)

connector.create_database("bagels")

connector.dbname = "bagels"

schemas = ("models", "data", "users")

[connector.create_schema(schema) for schema in schemas]

models_columns = {
    "id": "SERIAL PRIMARY KEY",
    "symbol": "TEXT",
    "trained_from": "DATE",
    "trained_to": "DATE",
    "n_training_points": "INTEGER",
    "model_summary": "TEXT",
    "date_trained": "DATE",
    "features": "TEXT",
    "loss": "REAL",
    "days_forecast": "INTEGER",
}

connector.create_table("models", models_columns, schema="models")

connector.add_unique_key(
    "models",
    [
        "date_trained",
        "features",
        "symbol",
        "days_forecast",
        "model_summary",
        "n_training_points",
    ],
    "models_unique_key",
    schema="models",
)

model_predictions_columns = {
    "id": "SERIAL PRIMARY KEY",
    "model_id": "INTEGER",
    "date": "DATE",
    "prediction": "REAL",
    "actual": "REAL",
}

connector.create_table("model_predictions", model_predictions_columns, schema="models")

connector.add_unique_key(
    "model_predictions",
    ["model_id", "date"],
    "model_predictions_unique_key",
    schema="models",
)
connector.add_foreign_key(
    "model_predictions", "model_id", "models", "id", schema="models"
)
