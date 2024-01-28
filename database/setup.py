# set up the database from scratch.
from database import PostgreSQLConnector

connector = PostgreSQLConnector(
    host="localhost",
    port="5432",
    user="postgres",
    password="password",
)

#connector.create_database("bagels")

connector.dbname = "bagels"

#schemas = ("spy", "schd", "qqq", "vym", "vtv")
schemas = ("agg", "vt")
[connector.create_schema(schema) for schema in schemas]

models_columns = {
    "id": "SERIAL PRIMARY KEY",
    "trained_on_date": "DATE",
    "features": "JSONB",
    "architecture": "JSONB",
    "hyperparameters": "JSONB",
    "training_loss_info": "JSONB",
}

[connector.create_table("models", models_columns, schema=schema) for schema in schemas]

[
    connector.add_unique_key(
        "models",
        [
            "trained_on_date",
            "features",
            "architecture",
            "hyperparameters",
        ],
        "models_unique_key",
        schema=schema,
    )
    for schema in schemas
]

model_predictions_columns = {
    "id": "SERIAL PRIMARY KEY",
    "model_id": "INTEGER",
    "date": "DATE",
    "predicted_price": "REAL",
    "prediction_made_on_date": "DATE",
}

[
    connector.create_table("forecasts", model_predictions_columns, schema=schema)
    for schema in schemas
]

[
    connector.add_unique_key(
        "forecasts",
        ["model_id", "date"],
        "model_predictions_unique_key",
        schema=schema,
    )
    for schema in schemas
]

[
    connector.add_foreign_key("forecasts", "model_id", "models", "id", schema=schema)
    for schema in schemas
]

training_data_columns = {
    "id": "SERIAL PRIMARY KEY",
    "model_id": "INTEGER",
    "data_blob": "BYTEA",
}

[
    connector.create_table("training_data", training_data_columns, schema=schema)
    for schema in schemas
]
[
    connector.create_table("prediction_data", training_data_columns, schema=schema)
    for schema in schemas
]

[
    connector.add_foreign_key(
        "training_data", "model_id", "models", "id", schema=schema
    )
    for schema in schemas
]

[
    connector.add_foreign_key(
        "prediction_data", "model_id", "models", "id", schema=schema
    )
    for schema in schemas
]

backtest_results_cols = {
    "id": "SERIAL PRIMARY KEY",
    "model_id": "INTEGER",
    "mean_absolute_percentage_error": "REAL",
    "classification_accuracy_percentage": "REAL",
    "number_of_training_windows": "INTEGER",
    "bootstrap_price_range": "TEXT",
    "mpae_price_range": "TEXT",
    "data_blob": "BYTEA",
}

[
    connector.create_table("backtest_results", backtest_results_cols, schema=schema)
    for schema in schemas
]

[
    connector.add_foreign_key(
        "backtest_results", "model_id", "models", "id", schema=schema
    )
    for schema in schemas
]
