from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.architecture import architecture_router
from api.routes.auth import auth_router
from api.routes.features import features_router
from api.routes.forecasts import forecasts_router
from api.routes.homepage import homepage_route
from api.routes.backtest_results import backtest_results_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(homepage_route)
app.include_router(forecasts_router, prefix="/forecasts")
app.include_router(architecture_router, prefix="/model_metadata")
app.include_router(features_router, prefix="/features")
app.include_router(backtest_results_router)
app.include_router(auth_router)
