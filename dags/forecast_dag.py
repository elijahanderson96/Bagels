from datetime import datetime

from airflow import DAG
from airflow.operators.email_operator import EmailOperator
from airflow.operators.python_operator import PythonOperator

from database import db_connector
from scripts import update_actual_values
from scripts.etf_predictor import ETFPredictor
from scripts.ingestion_fred import data_refresh
from scripts.ingestion_fred import endpoints
from scripts.ingestion_fred import etfs
from scripts.update_actual_values import update_actual_values

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email": ["elijahanderson96@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}

dag = DAG(
    "update_etf_dag",
    default_args=default_args,
    description="A simple tutorial DAG",
    schedule_interval="@daily",
    catchup=False,
)


def run_etf_predictor():
    api_key = "7f54d62f0a53c2b106b903fc80ecace1"
    tables = [endpoint.lower() for endpoint in endpoints.keys()]
    data_refresh(api_key)
    for etf in etfs:
        predictor = ETFPredictor(
            table_names=tables, etf_symbol=etf, days_forecast=28, connector=db_connector
        )
        predictor.predict()


def run_update_actual_values():
    update_actual_values(db_connector)


t1 = PythonOperator(
    task_id="run_etf_predictor",
    python_callable=run_etf_predictor,
    dag=dag,
)

t2 = PythonOperator(
    task_id="run_update_actual_values",
    python_callable=run_update_actual_values,
    dag=dag,
)

t3 = EmailOperator(
    task_id="send_email",
    to="elijahanderson96@gmail.com",
    subject="Airflow Alert",
    html_content="Your DAG has completed successfully!",
    dag=dag,
)

t1 >> t2 >> t3
