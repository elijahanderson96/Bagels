from fastapi import APIRouter, HTTPException
from typing import List
from database.database import db_connector  # Import your database connector

homepage_route = APIRouter()

@homepage_route.get("/etfs", response_model=List[str])
async def get_etfs():
    query = """
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('public', 'pg_catalog', 'users', 'information_schema', 'pg_toast');
    """

    try:
        etfs = db_connector.run_query(query)
        return etfs['schema_name'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
