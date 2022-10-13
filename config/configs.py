import os
from dotenv import load_dotenv
from config.filepaths import *

load_dotenv(PATH_TO_DOTENV, verbose=True, override=True)
print('Loading environment variables')

ENV = os.getenv('ENV')

if ENV == 'production':
    POSTGRES_URL = os.getenv('POSTGRES_PROD_URL')
    TOKEN = os.getenv('PRODUCTION_TOKEN')
    BASE_URL = os.getenv('PRODUCTION_IEX_URL')
else:
    POSTGRES_URL = os.getenv('POSTGRES_DEV_URL')
    TOKEN = os.getenv('SANDBOX_TOKEN')
    BASE_URL = os.getenv('SANDBOX_IEX_URL')

