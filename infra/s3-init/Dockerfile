FROM python:3.12-alpine

RUN pip install --no-cache-dir boto3
COPY create_bucket.py /create_bucket.py

ENTRYPOINT ["python", "/create_bucket.py"]
