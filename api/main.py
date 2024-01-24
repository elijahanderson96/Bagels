from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.forecasts import forecasts_router
from api.routes.architecture import architecture_router
from api.routes.features import features_router


app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins (use ["*"] for all origins)
    allow_credentials=True,
    allow_methods=["*"],  # List of allowed methods (use ["*"] for all methods)
    allow_headers=["*"],  # List of allowed headers (use ["*"] for all headers)
)

# Include the router
app.include_router(forecasts_router, prefix="/forecasts")

# These routers are to populate the tabs associated with an individual model, hence
# the prefix
app.include_router(architecture_router, prefix="/model_metadata")
app.include_router(features_router, prefix="/features")
