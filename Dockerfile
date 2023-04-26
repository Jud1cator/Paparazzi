FROM apache/airflow:2.5.0-python3.10
USER root
RUN apt update && apt install -y ffmpeg && rm -rf /var/lib/apt/lists/*
USER airflow
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /requirements.txt