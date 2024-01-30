from fastapi import APIRouter
from fastapi import HTTPException
from database.database import db_connector
from api.models.auth import UserLogin
from jose import jwt
from datetime import datetime, timedelta
import os
import bcrypt
import pandas as pd

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
auth_router = APIRouter()

@auth_router.post("/login")
async def login(user_credentials: UserLogin):
    username = user_credentials.username
    password = user_credentials.password

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # Update last_logged_in
    query = "UPDATE users.users SET last_logged_in = %s WHERE username = %s"
    db_connector.run_query(query, (datetime.now(), username), return_df=False)

    # Generate JWT token
    token = create_jwt_token(user['id'])

    return {"access_token": token, "token_type": "bearer"}

def authenticate_user(username: str, password: str):
    query = "SELECT * FROM users.users WHERE username = %s"
    result = db_connector.run_query(query, (username,))

    if not result.empty and verify_password(password, result.at[0, 'password_hash']):
        return result.iloc[0].to_dict()  # Convert the user row to a dict
    return None

def create_jwt_token(user_id: int):
    expire = datetime.utcnow() + timedelta(hours=8)  # Token expiration time
    to_encode = {"exp": expire, "sub": str(user_id)}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(input_password: str, stored_password_hash: str):
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_password_hash.encode('utf-8'))
