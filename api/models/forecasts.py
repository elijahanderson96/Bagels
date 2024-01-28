from typing import Optional

from pydantic import BaseModel



class ForecastsTable(BaseModel):
    model_id: int
    etf: str
    date: str
    trained_on_date: str
    predicted_price: float
    prediction_made_on_date: str
    mpae_price_range: str
    bootstrap_price_range: str
    mean_absolute_percentage_error: float
    classification_accuracy_percentage: float
    number_of_training_windows: int
    epochs: int
    stride: int
    overlap: int
    from_date: str
    batch_size: int
    learning_rate:float
    window_length: int
    sequence_length: int
    l1_regularization: Optional[float]
    l2_regularization: Optional[float]
