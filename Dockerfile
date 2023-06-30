# start from NVIDIA's TensorFlow GPU image
FROM tensorflow/tensorflow:latest-gpu

# set the working directory
WORKDIR /app

# copy the requirements file
COPY requirements.txt ./requirements.txt

# install the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./lstm.py /app/
COPY ./database.py /app/
COPY ./pipeline_metadata.yml /app

COPY ./one_time_sql_scripts /app/one_time_sql_scripts
# set the default command to execute the Python script
CMD ["python3", "lstm.py"]