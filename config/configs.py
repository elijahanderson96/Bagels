import os
from dotenv import load_dotenv
from config.filepaths import *

load_dotenv(PATH_TO_DOTENV, verbose=True, override=True)
print('Loading environment variables')

postgres_con = {
    'host': os.getenv('POSTGRES_HOST_ADDRESS'),
    'port': os.getenv('POSTGRES_PORT'),
    'dbname': os.getenv('POSTGRES_NAME'),
    'user': os.getenv('POSTGRES_USERNAME'),
    'password': os.getenv('POSTGRES_PASSWORD')
}


