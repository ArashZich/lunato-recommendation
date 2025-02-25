import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import uuid
from datetime import datetime

from api.utils.image_processing import base64_to_opencv, opencv_to_base64, resize_image, enhance_image
from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def detect_face_local(image_data: str) -> Dict[str, Any]:
    """
    Detect faces in the image locally (without using Celery tasks).
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Detection results with face coordinates
    """
    try:
        logger.info("Starting face detection")
        
        # Convert base64 to OpenCV image
        opencv_image = base64_to_opencv(image_data)
        
        if opencv_image is None:
            logger.error("Failed to convert image data")
            return {
                "success": False,
                "message": "Invalid image data",
                "faces": []
            }
            
        # Resize image if too large
        opencv_image = resize_image(opencv_image)
        
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
            logger.warning("No faces detected")
            return {
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
            
            # Generate a unique detection ID
            detection_id = str(uuid.uuid4())
            
            logger.info("Face detection completed")
            
            return {
                "success": True,
                "detection_id": detection_id,
                "face": face_data,
                "message": "Face detected successfully"
            }
            
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}",
            "faces": []
        }


async def get_face_region(image_data: str, face_coordinates: Dict[str, int], margin: float = 0.2) -> Optional[str]:
    """
    Extract face region from image with margin.
    
    Args:
        image_data: Base64 encoded image data
        face_coordinates: Dictionary with x, y, width, height
        margin: Margin around face as percentage of face size
        
    Returns:
        str: Base64 encoded face image or None if extraction fails
    """
    try:
        # Convert base64 to OpenCV image
        opencv_image = base64_to_opencv(image_data)
        
        if opencv_image is None:
            logger.error("Failed to convert image data")
            return None
            
        # Extract face coordinates
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['width'], face_coordinates['height']
        
        # Calculate margin in pixels
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate crop coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(opencv_image.shape[1], x + w + margin_x)
        y2 = min(opencv_image.shape[0], y + h + margin_y)
        
        # Crop image
        face_image = opencv_image[y1:y2, x1:x2]
        
        # Convert back to base64
        return opencv_to_base64(face_image)
        
    except Exception as e:
        logger.error(f"Error extracting face region: {str(e)}")
        return None


def get_face_landmarks(face_image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Extract facial landmarks (simplified version without dlib).
    
    Args:
        face_image: OpenCV face image
        
    Returns:
        list: Facial landmarks coordinates or None if detection fails
    """
    try:
        # This is a simplified version without dlib
        # In a real implementation, you would use dlib or another library to extract actual landmarks
        
        # Get image dimensions
        height, width = face_image.shape[:2]
        
        # Create synthetic landmarks (for testing purposes)
        # In a real implementation, these would come from dlib's shape_predictor
        landmarks = [
            # Jaw line
            (int(width * 0.1), int(height * 0.6)),  # 0
            (int(width * 0.2), int(height * 0.7)),  # 1
            (int(width * 0.3), int(height * 0.75)),  # 2
            (int(width * 0.4), int(height * 0.8)),  # 3
            (int(width * 0.5), int(height * 0.85)),  # 4
            (int(width * 0.6), int(height * 0.8)),  # 5
            (int(width * 0.7), int(height * 0.75)),  # 6
            (int(width * 0.8), int(height * 0.7)),  # 7
            (int(width * 0.9), int(height * 0.6)),  # 8
            
            # Right eyebrow
            (int(width * 0.3), int(height * 0.3)),  # 9
            (int(width * 0.35), int(height * 0.25)),  # 10
            (int(width * 0.4), int(height * 0.25)),  # 11
            
            # Left eyebrow
            (int(width * 0.6), int(height * 0.25)),  # 12
            (int(width * 0.65), int(height * 0.25)),  # 13
            (int(width * 0.7), int(height * 0.3)),  # 14
            
            # Nose
            (int(width * 0.5), int(height * 0.4)),  # 15
            (int(width * 0.5), int(height * 0.5)),  # 16
            (int(width * 0.45), int(height * 0.55)),  # 17
            (int(width * 0.5), int(height * 0.6)),  # 18
            (int(width * 0.55), int(height * 0.55)),  # 19
            
            # Right eye
            (int(width * 0.3), int(height * 0.35)),  # 20
            (int(width * 0.35), int(height * 0.33)),  # 21
            (int(width * 0.4), int(height * 0.35)),  # 22
            (int(width * 0.35), int(height * 0.37)),  # 23
            
            # Left eye
            (int(width * 0.6), int(height * 0.35)),  # 24
            (int(width * 0.65), int(height * 0.33)),  # 25
            (int(width * 0.7), int(height * 0.35)),  # 26
            (int(width * 0.65), int(height * 0.37)),  # 27
            
            # Mouth
            (int(width * 0.35), int(height * 0.7)),  # 28
            (int(width * 0.45), int(height * 0.7)),  # 29
            (int(width * 0.5), int(height * 0.7)),  # 30
            (int(width * 0.55), int(height * 0.7)),  # 31
            (int(width * 0.65), int(height * 0.7)),  # 32
            (int(width * 0.6), int(height * 0.72)),  # 33
            (int(width * 0.5), int(height * 0.73)),  # 34
            (int(width * 0.4), int(height * 0.72)),  # 35
        ]
        
        return landmarks
        
    except Exception as e:
        logger.error(f"Error extracting facial landmarks: {str(e)}")
        return None