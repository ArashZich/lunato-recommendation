import cv2
import numpy as np
import base64
import logging
from celery import shared_task
from pymongo import MongoClient
import io
from PIL import Image
from api.config import get_settings
from celery_app.app import app
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import dlib
import math

logger = logging.getLogger(__name__)
settings = get_settings()

# MongoDB connection for Celery tasks (synchronous)
mongo_client = MongoClient(settings.mongodb_uri)
db = mongo_client[settings.mongo_db_name]

# Facial shape categories
FACE_SHAPES = {
    'OVAL': 'بیضی',
    'ROUND': 'گرد',
    'SQUARE': 'مربعی',
    'HEART': 'قلبی',
    'OBLONG': 'کشیده',
    'DIAMOND': 'لوزی',
    'TRIANGLE': 'مثلثی'
}


@shared_task(name="celery_app.tasks.face_analysis.analyze_face_shape")
def analyze_face_shape(detection_id, image_data, user_id, metadata):
    """
    Analyze the detected face to determine face shape.
    
    Args:
        detection_id (str): ID of the face detection record
        image_data (str): Base64 encoded image data
        user_id (str): User ID for tracking
        metadata (dict): Additional metadata
        
    Returns:
        dict: Analysis results with face shape
    """
    try:
        logger.info(f"Starting face shape analysis for detection {detection_id}")
        
        # Get the face detection record
        detection = db.face_detections.find_one({"_id": detection_id})
        if not detection:
            logger.error(f"Face detection record {detection_id} not found")
            return {
                "success": False,
                "message": "Face detection record not found"
            }
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract face coordinates
        face_coord = detection["face_coordinates"]
        x, y, w, h = face_coord["x"], face_coord["y"], face_coord["width"], face_coord["height"]
        
        # Add some margin to the face region
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(opencv_image.shape[1], x + w + margin_x)
        y2 = min(opencv_image.shape[0], y + h + margin_y)
        
        # Extract face region
        face_image = opencv_image[y1:y2, x1:x2]
        
        # Get facial landmarks
        landmarks = get_facial_landmarks(face_image)
        
        if not landmarks:
            logger.warning(f"Could not detect facial landmarks for detection {detection_id}")
            return {
                "success": False,
                "message": "Could not detect facial landmarks"
            }
        
        # Extract face shape features
        features = extract_face_shape_features(landmarks)
        
        # Determine face shape
        face_shape, confidence, shape_metrics = classify_face_shape(features)
        
        # Save analysis results to database
        analysis_record = {
            "detection_id": detection_id,
            "user_id": user_id,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_metrics": shape_metrics,
            "features": features,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "status": "completed"
        }
        
        db.face_analyses.insert_one(analysis_record)
        
        # Update the detection record
        db.face_detections.update_one(
            {"_id": detection_id},
            {"$set": {"status": "analyzed", "face_shape": face_shape}}
        )
        
        logger.info(f"Face shape analysis completed for detection {detection_id}: {face_shape}")
        
        # Queue the next task for frame matching
        from celery_app.tasks.frame_matching import match_frames
        match_frames.delay(detection_id, face_shape, user_id, metadata)
        
        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_metrics": shape_metrics,
            "next_task": "frame_matching"
        }
        
    except Exception as e:
        logger.error(f"Error in face shape analysis: {str(e)}")
        
        # Update the detection record with error status
        if detection_id:
            db.face_detections.update_one(
                {"_id": detection_id},
                {"$set": {"status": "error", "error_message": str(e)}}
            )
            
        return {
            "success": False,
            "message": f"Error analyzing face shape: {str(e)}"
        }


def get_facial_landmarks(face_image):
    """
    Extract facial landmarks using dlib's landmark detector
    
    Args:
        face_image: OpenCV face image
        
    Returns:
        list: Facial landmarks coordinates
    """
    try:
        # Initialize dlib's face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if not faces:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Predict facial landmarks
        shape = predictor(gray, face)
        
        # Convert landmarks to numpy array
        landmarks = []
        for i in range(68):
            landmarks.append((shape.part(i).x, shape.part(i).y))
            
        return np.array(landmarks)
        
    except Exception as e:
        logger.error(f"Error detecting facial landmarks: {str(e)}")
        return None


def extract_face_shape_features(landmarks):
    """
    Extract features from facial landmarks that help determine face shape
    
    Args:
        landmarks: Numpy array of facial landmarks
        
    Returns:
        dict: Features relevant to face shape classification
    """
    # Calculate key measurements
    # Forehead width (points 0-16)
    forehead_width = np.linalg.norm(landmarks[0] - landmarks[16])
    
    # Cheekbone width (points 1-15)
    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])
    
    # Jawline width (points 3-13)
    jawline_width = np.linalg.norm(landmarks[3] - landmarks[13])
    
    # Jaw angle points (points 4-12)
    jaw_angle_width = np.linalg.norm(landmarks[4] - landmarks[12])
    
    # Face length (middle of chin to middle of forehead)
    chin_point = landmarks[8]
    forehead_point = landmarks[27]  # Between eyebrows
    face_length = np.linalg.norm(chin_point - forehead_point)
    
    # Jawline length (average of left and right sides)
    left_jawline = np.sum([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(3, 7)])
    right_jawline = np.sum([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(9, 13)])
    jawline_length = (left_jawline + right_jawline) / 2
    
    # Calculate key ratios
    width_to_length_ratio = cheekbone_width / face_length
    cheekbone_to_jaw_ratio = cheekbone_width / jawline_width
    forehead_to_cheekbone_ratio = forehead_width / cheekbone_width
    
    # Jaw angle (in degrees)
    chin_to_jaw_left_vec = landmarks[3] - landmarks[8]
    chin_to_jaw_right_vec = landmarks[13] - landmarks[8]
    
    cos_angle = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec) / (
        np.linalg.norm(chin_to_jaw_left_vec) * np.linalg.norm(chin_to_jaw_right_vec)
    )
    jaw_angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    
    return {
        'width_to_length_ratio': float(width_to_length_ratio),
        'cheekbone_to_jaw_ratio': float(cheekbone_to_jaw_ratio),
        'forehead_to_cheekbone_ratio': float(forehead_to_cheekbone_ratio),
        'jaw_angle': float(jaw_angle),
        'forehead_width': float(forehead_width),
        'cheekbone_width': float(cheekbone_width),
        'jawline_width': float(jawline_width),
        'face_length': float(face_length),
        'jawline_length': float(jawline_length),
    }


def classify_face_shape(features):
    """
    Classify the face shape based on extracted features
    
    Args:
        features: Dict of facial measurements and ratios
        
    Returns:
        tuple: (face_shape, confidence, shape_metrics)
    """
    # Extract features
    width_to_length = features['width_to_length_ratio']
    cheekbone_to_jaw = features['cheekbone_to_jaw_ratio']
    forehead_to_cheekbone = features['forehead_to_cheekbone_ratio']
    jaw_angle = features['jaw_angle']
    
    # Initialize shape scores
    shape_scores = {
        'OVAL': 0,
        'ROUND': 0,
        'SQUARE': 0,
        'HEART': 0,
        'OBLONG': 0,
        'DIAMOND': 0,
        'TRIANGLE': 0
    }
    
    # Evaluate each shape based on the features
    
    # Oval face shape (balanced proportions)
    # Width to length around 0.75, smooth jaw
    if 0.7 <= width_to_length <= 0.8:
        shape_scores['OVAL'] += 3
    if 1.0 <= cheekbone_to_jaw <= 1.1:
        shape_scores['OVAL'] += 2
    if 120 <= jaw_angle <= 140:
        shape_scores['OVAL'] += 1
    
    # Round face shape (width similar to length)
    # Width to length close to 1, wider cheeks, soft jaw
    if 0.9 <= width_to_length <= 1.1:
        shape_scores['ROUND'] += 3
    if 1.1 <= cheekbone_to_jaw <= 1.3:
        shape_scores['ROUND'] += 2
    if jaw_angle >= 130:
        shape_scores['ROUND'] += 1
    
    # Square face shape (angular jaw, width similar to length)
    # Width to length around 0.9-1.0, jaw width close to cheekbone
    if 0.9 <= width_to_length <= 1.0:
        shape_scores['SQUARE'] += 2
    if 0.9 <= cheekbone_to_jaw <= 1.05:
        shape_scores['SQUARE'] += 3
    if jaw_angle <= 125:
        shape_scores['SQUARE'] += 2
    
    # Heart face shape (wider forehead, narrower jaw)
    # Forehead wider than cheekbones, narrow jaw
    if forehead_to_cheekbone >= 1.0:
        shape_scores['HEART'] += 3
    if cheekbone_to_jaw >= 1.3:
        shape_scores['HEART'] += 2
    
    # Oblong face shape (long face)
    # Width to length ratio small
    if width_to_length <= 0.7:
        shape_scores['OBLONG'] += 3
    if 0.9 <= cheekbone_to_jaw <= 1.1:
        shape_scores['OBLONG'] += 1
    
    # Diamond face shape (narrow forehead and jaw, wide cheekbones)
    # Forehead narrower than cheekbones, jaw narrower than cheekbones
    if forehead_to_cheekbone <= 0.9:
        shape_scores['DIAMOND'] += 2
    if cheekbone_to_jaw >= 1.2:
        shape_scores['DIAMOND'] += 2
    if jaw_angle >= 130:
        shape_scores['DIAMOND'] += 1
    
    # Triangle face shape (narrow forehead, wide jaw)
    # Forehead narrower than cheekbones, wide jaw
    if forehead_to_cheekbone <= 0.85:
        shape_scores['TRIANGLE'] += 2
    if cheekbone_to_jaw <= 0.9:
        shape_scores['TRIANGLE'] += 3
    
    # Find the face shape with the highest score
    face_shape = max(shape_scores, key=shape_scores.get)
    max_score = shape_scores[face_shape]
    
    # Calculate confidence as a percentage of the maximum possible score
    # (maximum score would be around 5-6 for most shapes)
    confidence = min(1.0, max_score / 6) * 100
    
    return face_shape, confidence, shape_scores