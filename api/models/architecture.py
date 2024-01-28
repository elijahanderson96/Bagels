from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel


class NeuralNetworkArchitecture(BaseModel):
    layers: List[Dict]
    hyperparameters: Optional[Dict[str, Any]] = None
    training_loss_info: Optional[Dict[str, Any]] = None
