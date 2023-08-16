FROM apache/airflow:latest-python3.11

USER root
RUN apt-get update
RUN chown -R airflow: /opt/airflow

WORKDIR /opt/airflow

USER airflow
COPY . /opt/airflow/
ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow/"

RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /opt/airflow/requirements.txt



