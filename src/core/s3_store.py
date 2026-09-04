# src/core/s3_store.py

import os
import tarfile
import boto3
import logging

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "industrial-vic-copilot-674963338816-us-east-1-an")
LOCAL_DATA_DIR = "data"

VECTORSTORE_S3_KEY = "chroma_index.tar.gz"
VECTORSTORE_PATH = os.path.join(LOCAL_DATA_DIR, "vectorstore")

RAW_DOCS_S3_KEY = "raw_docs.tar.gz"
RAW_DOCS_PATH = os.path.join(LOCAL_DATA_DIR, "raw")

def _hydrate_from_s3(s3_key: str, target_path: str, label: str):
    """Generic helper: download + extract a tarball from S3 if target_path is empty."""
    if os.path.exists(target_path) and os.listdir(target_path):
        logger.info(f"{label} already present at {target_path}, skipping S3 download.")
        return

    local_tar_path = f"/tmp/{s3_key}"
    logger.info(f"Downloading {label} from s3://{S3_BUCKET}/{s3_key} ...")

    try:
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, s3_key, local_tar_path)
        logger.info(f"{label} download complete. Extracting...")

        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=LOCAL_DATA_DIR)

        os.remove(local_tar_path)
        logger.info(f"{label} hydrated successfully at {target_path}")

    except Exception as e:
        logger.error(f"Failed to hydrate {label} from S3: {e}")
        raise

def hydrate_vectorstore_from_s3():
    _hydrate_from_s3(VECTORSTORE_S3_KEY, VECTORSTORE_PATH, "Vector store")

def hydrate_raw_docs_from_s3():
    _hydrate_from_s3(RAW_DOCS_S3_KEY, RAW_DOCS_PATH, "Raw documents")

