# start from NVIDIA's TensorFlow GPU image
FROM tensorflow/tensorflow:latest-gpu

# set the working directory
WORKDIR /app

# copy the requirements file
COPY requirements.txt ./requirements.txt

# install the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./etf_predictor.py /app/
COPY ./database.py /app/
COPY ./pipeline_metadata.yml /app

COPY ./scripts /app/scripts
# set the default command to execute the Python script
CMD ["python3", "etf_predictor.py"]
