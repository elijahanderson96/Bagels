# Use official base image of Python
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install the cronjob and Airflow
RUN apt-get update && apt-get install -y cron && pip install apache-airflow

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

