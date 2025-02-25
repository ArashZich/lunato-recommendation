"""
Celery configuration settings
"""
from api.config import get_settings

settings = get_settings()

# Broker settings
broker_url = settings.celery_broker_url
result_backend = settings.celery_result_backend

# Serialization settings
task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"

# Task settings
task_track_started = True
task_time_limit = 600  # 10 minutes
worker_prefetch_multiplier = 1

# Timezone settings
timezone = "UTC"
enable_utc = True

# Queue settings - route tasks to specific queues
task_routes = {
    "celery_app.tasks.face_detection.*": {"queue": "face_detection"},
    "celery_app.tasks.face_analysis.*": {"queue": "face_analysis"},
    "celery_app.tasks.frame_matching.*": {"queue": "frame_matching"},
}

# Worker concurrency settings
worker_concurrency = 1  # Set based on available CPU cores

# Logging
worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s"