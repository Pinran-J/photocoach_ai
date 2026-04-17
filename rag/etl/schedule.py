"""
ETL – Scheduler
Runs the ingestion pipeline on a recurring schedule using APScheduler.

Usage (local / dev):
    python -m rag.etl.schedule

AWS Production alternative:
    Deploy run_pipeline() as an AWS Lambda function and trigger it with
    an EventBridge Scheduler rule (e.g. "cron(0 2 ? * MON *)").
    Store PINECONE_API_KEY and OPENAI_API_KEY in AWS Secrets Manager and
    inject them as Lambda environment variables.
"""

import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rag.etl.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def scheduled_job() -> None:
    logger.info("Scheduled ETL job triggered.")
    try:
        summary = run_pipeline()
        logger.info("Scheduled run finished: %s", summary)
    except Exception as e:
        logger.error("Scheduled run failed: %s", e, exc_info=True)


def start_scheduler(
    day_of_week: str = "mon",
    hour: int = 2,
    minute: int = 0,
) -> None:
    """
    Start the blocking scheduler.
    Default: every Monday at 02:00 local time.
    Adjust day_of_week / hour / minute to match your preferred cadence.
    """
    scheduler = BlockingScheduler()
    trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
    scheduler.add_job(scheduled_job, trigger, id="photocoach_etl", replace_existing=True)

    logger.info(
        "Scheduler started — ETL will run every %s at %02d:%02d. Press Ctrl+C to stop.",
        day_of_week,
        hour,
        minute,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()
