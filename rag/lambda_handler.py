"""
AWS Lambda handler for the PhotoCoach ETL pipeline.

Triggered by an EventBridge Scheduler rule (e.g. every Monday at 02:00 UTC).
Runs the full Extract → Transform → Load pipeline and returns a summary.

──────────────────────────────────────────────────────────────────────────────
DEPLOYMENT STEPS
──────────────────────────────────────────────────────────────────────────────

1. Build a Linux-compatible package (must use --platform flag — building on
   macOS produces macOS wheels that crash on Lambda's Linux runtime):

    rm -rf lambda_package etl_lambda.zip
    mkdir -p lambda_package

    pip install \
        --platform manylinux2014_x86_64 \
        --target lambda_package/ \
        --implementation cp \
        --python-version 3.12 \
        --only-binary=:all: \
        --upgrade \
        -r requirements-etl.txt

    mkdir -p lambda_package/rag/etl
    cp rag/etl/__init__.py   lambda_package/rag/etl/
    cp rag/etl/extractors.py lambda_package/rag/etl/
    cp rag/etl/transform.py  lambda_package/rag/etl/
    cp rag/etl/load.py       lambda_package/rag/etl/
    cp rag/etl/pipeline.py   lambda_package/rag/etl/
    cp rag/lambda_handler.py lambda_package/lambda_handler.py

    cd lambda_package && zip -r ../etl_lambda.zip . -x "*.pyc" -x "__pycache__/*" && cd ..

2. Upload to Lambda (AWS Console):

    - Runtime:      Python 3.12
    - Architecture: x86_64
    - Handler:      lambda_handler.handler
    - Timeout:      15 min 0 sec
    - Memory:       512 MB
    - Upload:       Code tab → Upload from → .zip file → etl_lambda.zip

3. Set environment variables (do NOT use a .env file):

    Configuration → Environment variables → Edit → Add:
        OPENAI_API_KEY   = sk-...
        PINECONE_API_KEY = pcsk_...

4. Schedule with EventBridge (AWS Console):

    Lambda → Configuration → Triggers → Add trigger
    → Source: EventBridge (CloudWatch Events)
    → Create a new rule
        Rule name:   photocoach-etl-weekly
        Rule type:   Schedule expression
        Expression:  cron(0 2 ? * MON *)
    → Add

    This fires every Monday at 02:00 UTC, ingesting fresh RSS articles
    and re-syncing reference pages into the Pinecone index.

──────────────────────────────────────────────────────────────────────────────
NOTES
──────────────────────────────────────────────────────────────────────────────

- The Pinecone SDK's upsert uses a ThreadPool internally which requires POSIX
  semaphores unavailable in Lambda. load.py bypasses this by calling the
  Pinecone REST API directly via requests instead.

- PDF extraction is skipped in Lambda (no local filesystem). Only web reference
  pages and RSS feeds are ingested. To include PDFs, store them in S3 and
  download to /tmp at runtime.

- lambda_package/ and etl_lambda.zip are build artifacts — both are gitignored
  and should never be committed.

──────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict, context) -> dict:
    """Lambda entry point.

    EventBridge passes a scheduled event dict; we ignore its contents and just
    run the pipeline. The function returns a summary dict as the response body.
    """
    logger.info("PhotoCoach ETL Lambda triggered. Event: %s", json.dumps(event))

    # Import here (not at module level) so cold-start logging captures any
    # import errors before the handler is invoked.
    from rag.etl.pipeline import run_pipeline

    try:
        summary = run_pipeline()
        logger.info("ETL pipeline succeeded: %s", summary)
        return {
            "statusCode": 200,
            "body": json.dumps(summary),
        }
    except Exception as exc:
        logger.error("ETL pipeline failed: %s", str(exc), exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc)}),
        }
