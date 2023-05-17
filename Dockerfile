FROM python:3.9
WORKDIR /src
ENV PYTHONPATH="${PYTHONPATH}:/src"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
