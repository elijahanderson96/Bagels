from typing import Dict
from typing import List

from pydantic import BaseModel


class NeuralNetworkArchitecture(BaseModel):
    name: str
    layers: List[Dict]
