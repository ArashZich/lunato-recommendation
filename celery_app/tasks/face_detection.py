import cv2
import numpy as np
import base64
import logging
from celery import shared_task
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import io
from PIL import Image
from api.config import get_settings
from celery_app.app import app
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)
settings = get_settings()

# MongoDB connection for Celery tasks (synchronous)
mongo_client = MongoClient(settings.mongodb_uri)
db = mongo_client[settings.mongo_db_name]


@shared_task(name="celery_app.tasks.face_detection.detect_face")
def detect_face(image_data, user_id, metadata):
    """
    Detect faces in the image and save the results to the database.
    
    Args:
        image_data (str): Base64 encoded image data
        user_id (str): User ID for tracking
        metadata (dict): Additional metadata like device info
        
    Returns:
        dict: Detection results with face coordinates
    """
    try:
        logger.info(f"Starting face detection for user {user_id}")
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load face detector
        face_cascade_path = cv2.data.haarcascades + settings.face_detection_model
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process detection results
        if len(faces) == 0:
            logger.warning(f"No faces detected for user {user_id}")
            result = {
                "success": False,
                "message": "No faces detected in the image",
                "faces": []
            }
        else:
            # We'll focus on the largest face if multiple are detected
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Create face data
            face_data = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(x + w/2),
                "center_y": int(y + h/2),
                "aspect_ratio": float(w / h),
            }
            
            # Save face detection result to database
            detection_id = str(uuid.uuid4())
            detection_record = {
                "_id": detection_id,
                "user_id": user_id,
                "face_coordinates": face_data,
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "status": "detected",
                "image_width": opencv_image.shape[1],
                "image_height": opencv_image.shape[0]
            }
            
            # Insert into MongoDB
            db.face_detections.insert_one(detection_record)
            
            logger.info(f"Face detection completed for user {user_id}")
            
            result = {
                "success": True,
                "detection_id": detection_id,
                "face": face_data,
                "next_task": "face_analysis"
            }
            
            # Queue the next task for face analysis
            from celery_app.tasks.face_analysis import analyze_face_shape
            analyze_face_shape.delay(detection_id, image_data, user_id, metadata)
            
        return result
        
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}",
            "faces": []
        }


@shared_task(name="celery_app.tasks.face_detection.preprocess_image")
def preprocess_image(image_data):
    """
    Preprocess the image for better face detection
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        str: Preprocessed image data in base64
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if image is too large
        max_size = 1024
        h, w = opencv_image.shape[:2]
        
        if max(h, w) > max_size:
            scaling_factor = max_size / max(h, w)
            opencv_image = cv2.resize(
                opencv_image, 
                (int(w * scaling_factor), int(h * scaling_factor)), 
                interpolation=cv2.INTER_AREA
            )
        
        # Apply some light preprocessing
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)
        
        # Convert back to color
        processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        preprocessed_data = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "preprocessed_image": preprocessed_data
        }
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return {
            "success": False,
            "message": f"Error preprocessing image: {str(e)}",
            "preprocessed_image": None
        }