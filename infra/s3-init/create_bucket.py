import os
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

client = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

for _ in range(30):
    try:
        client.create_bucket(Bucket="mlflow")
        print("Bucket mlflow created")
        break
    except ClientError as error:
        if error.response["Error"]["Code"] in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
            print("Bucket mlflow already exists")
            break
        raise
    except EndpointConnectionError:
        time.sleep(2)
else:
    raise RuntimeError("MinIO did not become available")
