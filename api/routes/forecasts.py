from typing import List

from fastapi import APIRouter
from fastapi import HTTPException

from api.models.forecasts import ForecastsTable
from database.database import db_connector

forecasts_router = APIRouter()


@forecasts_router.get("/{etf_name}", response_model=List[ForecastsTable])
async def get_models(etf_name: str):
    #{"epochs": 25, "stride": 1, "overlap": 505, "from_date": "2020-01-01", "batch_size": 64, "learning_rate": 0.003, "window_length": 512, "sequence_length": 16, "l1_regularization": 0.12, "l2_regularization": 0.02}
    etf_name = etf_name.lower()
    # Replace with actual database query logic
    data = db_connector.run_query(
        f"""SELECT 
    m.id as model_id, 
    m.trained_on_date, 
    m.hyperparameters->>'epochs' as epochs,  
    m.hyperparameters->>'stride' as stride,  
    m.hyperparameters->>'overlap' as overlap,  
    m.hyperparameters->>'from_date' as from_date,  
    m.hyperparameters->>'batch_size' as batch_size,  
    m.hyperparameters->>'learning_rate' as learning_rate,  
    m.hyperparameters->>'window_length' as window_length,  
    m.hyperparameters->>'sequence_length' as sequence_length,  
    m.hyperparameters->>'l1_regularization' as l1_regularization, 
    m.hyperparameters->>'l2_regularization' as l2_regularization,      
    b.mean_absolute_percentage_error, 
    b.classification_accuracy_percentage, 
    b.number_of_training_windows, 
    b.bootstrap_price_range, 
    b.mpae_price_range,
    f.date, 
    f.predicted_price, 
    f.prediction_made_on_date
FROM 
    {etf_name}.models m
JOIN 
    {etf_name}.backtest_results b ON m.id = b.model_id
JOIN 
    {etf_name}.forecasts f ON m.id = f.model_id
"""
    )
    data["etf"] = etf_name

    for column in ["trained_on_date", "date", "prediction_made_on_date"]:
        data[column] = data[column].apply(lambda x: x.strftime("%Y-%m-%d"))

    if data.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    return data.to_dict("records")
