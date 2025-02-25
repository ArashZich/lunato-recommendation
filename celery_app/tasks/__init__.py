# Import Celery tasks to make them available via the celery_app.tasks namespace
from celery_app.tasks.face_detection import detect_face, preprocess_image
from celery_app.tasks.face_analysis import analyze_face_shape
from celery_app.tasks.frame_matching import match_frames