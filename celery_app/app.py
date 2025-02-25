from celery import Celery
import os
from api.config import get_settings

settings = get_settings()

# Initialize Celery
app = Celery(
    "eyeglass_recommendation",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "celery_app.tasks.face_detection",
        "celery_app.tasks.face_analysis",
        "celery_app.tasks.frame_matching"
    ]
)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    worker_prefetch_multiplier=1,
    task_routes={
        "celery_app.tasks.face_detection.*": {"queue": "face_detection"},
        "celery_app.tasks.face_analysis.*": {"queue": "face_analysis"},
        "celery_app.tasks.frame_matching.*": {"queue": "frame_matching"},
    },
)

# This ensures tasks are not executed synchronously in development/debug mode
app.conf.update(task_always_eager=False)

if __name__ == "__main__":
    app.start()