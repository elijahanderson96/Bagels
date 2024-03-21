from typing import Dict
from typing import List

from fastapi import APIRouter
from fastapi import HTTPException

from api.helpers import decompress_dataframe
from database.database import db_connector

backtest_results_router = APIRouter()


@backtest_results_router.get(
    "/backtest_results/{etf_name}/{model_id}", response_model=List[Dict]
)
async def get_backtest_results(etf_name: str, model_id: int):
    query = f"""
    SELECT data_blob FROM {etf_name.lower()}.backtest_results WHERE model_id = {model_id}
    """

    try:
        result = db_connector.run_query(query)["data_blob"].squeeze()

        if not result:
            raise HTTPException(status_code=404, detail="Data not found")

        df = decompress_dataframe(result)

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
