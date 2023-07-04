from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime
from scripts import etf_predictor, update_actual_values

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['elijahanderson96@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'update_etf_dag',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval='@daily',
)

def run_etf_predictor():
    etf_predictor.run()

def run_update_actual_values():
    update_actual_values.run()

t1 = PythonOperator(
    task_id='run_etf_predictor',
    python_callable=run_etf_predictor,
    dag=dag,
)

t2 = PythonOperator(
    task_id='run_update_actual_values',
    python_callable=run_update_actual_values,
    dag=dag,
)

t3 = EmailOperator(
    task_id='send_email',
    to='elijahanderson96@gmail.com',
    subject='Airflow Alert',
    html_content='Your DAG has completed successfully.',
    dag=dag
)

t1 >> t2 >> t3
