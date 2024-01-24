from pydantic import BaseModel


class ForecastsTable(BaseModel):
    model_id: int
    etf: str
    date: str
    trained_on_date: str
    predicted_price: float
    prediction_made_on_date: str
    # previousClose: float
    mpae_price_range: str
    bootstrap_price_range: str
    # percentageDiff: float
    mean_absolute_percentage_error: float
    classification_accuracy_percentage: float
    number_of_training_windows: int
