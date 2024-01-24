from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FeatureModel(BaseModel):
    # Define the structure of a single feature.
    # Update these fields according to your actual feature data structure.
    name: str
    description: str
    # Add more fields as needed.

class DataModel(BaseModel):
    # Since DataFrame rows are converted to dictionaries, use a generic Dict type.
    # If the structure of data rows is known and consistent, you can define it more precisely.
    data: List[Dict[str, Any]]

class FeaturesAndDataResponse(BaseModel):
    features: List[Dict[str, str]]
    training_data: DataModel
    prediction_data: DataModel
