from typing import Any

from pydantic import BaseModel
from typing import List, Dict

class FeatureDict(BaseModel):
    features: Dict[str, str]

class DataModel(BaseModel):
    data: List[Dict[str, Any]]

class FeaturesAndDataResponse(BaseModel):
    features: List[FeatureDict]
    training_data: DataModel
    prediction_data: DataModel
