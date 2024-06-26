import json

from fastapi import APIRouter
from fastapi import HTTPException

from api.models.architecture import NeuralNetworkArchitecture
from database.database import db_connector

architecture_router = APIRouter()


@architecture_router.get(
    "/{etf_name}/{model_id}", response_model=NeuralNetworkArchitecture
)
async def get_architecture(etf_name: str, model_id: int):
    etf_name = etf_name.lower()
    # Replace with actual database query logic to fetch neural network architecture
    architecture_data = db_connector.run_query(
        f"""SELECT architecture, hyperparameters, training_loss_info
            FROM {etf_name}.models
            WHERE id = {model_id}
        """
    )

    if architecture_data.empty:
        raise HTTPException(status_code=404, detail="Model architecture not found")

    response = {
        "layers": [architecture_data["architecture"].iloc[0]],
        "hyperparameters": architecture_data["hyperparameters"].iloc[0],
        "training_loss_info": architecture_data["training_loss_info"].iloc[0],
    }

    # Assuming the architecture data is stored in a specific format in the database
    # You'll need to parse or convert it into the NeuralNetworkArchitecture format
    nn_architecture = NeuralNetworkArchitecture(**response)

    return nn_architecture
