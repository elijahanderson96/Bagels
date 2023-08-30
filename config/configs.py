import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), 'config', '.env'), verbose=False, override=True)
print("Loading environment variables")

ENV = os.getenv("ENV")

print(f"We are in a {ENV} environment.")

db_config = {
    'host': os.getenv('POSTGRES_HOST_ADDRESS'),
    'port': os.getenv('POSTGRES_PORT'),
    'user': os.getenv('POSTGRES_USERNAME'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'dbname': os.getenv('POSTGRES_NAME')
}
