from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.forecasts import forecasts_router

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
