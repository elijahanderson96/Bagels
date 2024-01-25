from fastapi import APIRouter
from fastapi import HTTPException

from api.helpers import decompress_dataframe
from database.database import db_connector
from api.models.features import FeaturesAndDataResponse

features_router = APIRouter()


@features_router.get("/{etf_name}/{model_id}", response_model=FeaturesAndDataResponse)
async def get_features_and_data(etf_name: str, model_id: int) -> FeaturesAndDataResponse:
    etf_name = etf_name.lower()

    features_query = f"SELECT features FROM {etf_name}.models WHERE id = {model_id}"
    features_data = db_connector.run_query(features_query)

    if features_data.empty:
        raise HTTPException(status_code=404, detail="Features not found")

    training_data_query = (
        f"SELECT data_blob FROM {etf_name}.training_data WHERE model_id = {model_id}"
    )
    prediction_data_query = (
        f"SELECT data_blob FROM {etf_name}.prediction_data WHERE model_id = {model_id}"
    )

    training_data_blob = db_connector.run_query(training_data_query).squeeze()
    prediction_data_blob = db_connector.run_query(prediction_data_query).squeeze()

    if not training_data_blob or not prediction_data_blob:
        raise HTTPException(status_code=404, detail="Data not found")

    training_df = decompress_dataframe(training_data_blob)
    prediction_df = decompress_dataframe(prediction_data_blob)
    # make features a dict
    return FeaturesAndDataResponse(
        features=features_data.to_dict("records"),
        training_data={"data": training_df.to_dict("records")},
        prediction_data={"data": prediction_df.to_dict("records")}
    )


