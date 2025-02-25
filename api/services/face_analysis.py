import numpy as np
import logging
import math
from typing import Dict, Any, List, Tuple, Optional

from api.utils.image_processing import base64_to_opencv, opencv_to_base64
from api.services.face_detection import get_face_landmarks

logger = logging.getLogger(__name__)

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


async def analyze_face_shape(face_image_data: str) -> Dict[str, Any]:
    """
    Analyze face shape from image.
    
    Args:
        face_image_data: Base64 encoded face image
        
    Returns:
        dict: Analysis results with face shape
    """
    try:
        logger.info("Starting face shape analysis")
        
        # Convert base64 to OpenCV image
        face_image = base64_to_opencv(face_image_data)
        
        if face_image is None:
            logger.error("Failed to convert image data")
            return {
                "success": False,
                "message": "Invalid image data"
            }
            
        # Get facial landmarks
        landmarks = get_face_landmarks(face_image)
        
        if not landmarks:
            logger.warning("Could not detect facial landmarks")
            return {
                "success": False,
                "message": "Could not detect facial landmarks"
            }
            
        # Extract face shape features
        features = extract_face_shape_features(landmarks)
        
        # Determine face shape
        face_shape, confidence, shape_metrics = classify_face_shape(features)
        
        logger.info(f"Face shape analysis completed: {face_shape}")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_metrics": shape_metrics
        }
        
    except Exception as e:
        logger.error(f"Error in face shape analysis: {str(e)}")
        return {
            "success": False,
            "message": f"Error analyzing face shape: {str(e)}"
        }


def extract_face_shape_features(landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Extract features from facial landmarks that help determine face shape
    
    Args:
        landmarks: List of facial landmarks (x, y) coordinates
        
    Returns:
        dict: Features relevant to face shape classification
    """
    # Convert landmarks to numpy array for easier calculations
    landmarks_np = np.array(landmarks)
    
    # Calculate key measurements
    # Note: These indices are approximations and would need to be adjusted
    # based on the actual landmark detection system used
    
    # Forehead width (distance between temples)
    forehead_width = np.linalg.norm(landmarks[0] - landmarks[8])
    
    # Cheekbone width (distance between cheekbones)
    cheekbone_width = np.linalg.norm(landmarks[2] - landmarks[6])
    
    # Jawline width (distance between jaw angles)
    jawline_width = np.linalg.norm(landmarks[3] - landmarks[5])
    
    # Face length (middle of chin to middle of forehead)
    chin_point = landmarks[4]  # Center of chin
    forehead_point = (landmarks[0] + landmarks[8]) // 2  # Middle of forehead
    face_length = np.linalg.norm(chin_point - forehead_point)
    
    # Jawline length (average of left and right sides)
    left_jawline = np.sum([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(0, 3)])
    right_jawline = np.sum([np.linalg.norm(landmarks[i] - landmarks[i+1]) for i in range(5, 8)])
    jawline_length = (left_jawline + right_jawline) / 2
    
    # Calculate key ratios
    width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
    cheekbone_to_jaw_ratio = cheekbone_width / jawline_width if jawline_width > 0 else 0
    forehead_to_cheekbone_ratio = forehead_width / cheekbone_width if cheekbone_width > 0 else 0
    
    # Jaw angle (in degrees)
    chin_to_jaw_left_vec = landmarks[3] - landmarks[4]
    chin_to_jaw_right_vec = landmarks[5] - landmarks[4]
    
    # Calculate cosine similarity between vectors
    if np.linalg.norm(chin_to_jaw_left_vec) > 0 and np.linalg.norm(chin_to_jaw_right_vec) > 0:
        cos_angle = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec) / (
            np.linalg.norm(chin_to_jaw_left_vec) * np.linalg.norm(chin_to_jaw_right_vec)
        )
        # Ensure the value is in the valid range for arccos
        cos_angle = min(1.0, max(-1.0, cos_angle))
        jaw_angle = math.degrees(math.acos(cos_angle))
    else:
        jaw_angle = 0.0
    
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


def classify_face_shape(features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
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


def get_face_shape_description(face_shape: str) -> str:
    """
    Get a description of the face shape and recommended frames.
    
    Args:
        face_shape: Face shape (OVAL, ROUND, etc.)
        
    Returns:
        str: Description of face shape and frame recommendations
    """
    descriptions = {
        'OVAL': """
            صورت بیضی متعادل‌ترین شکل صورت است. پهنای گونه‌ها با پهنای پیشانی و فک متناسب است.
            پیشنهاد: اکثر فریم‌ها برای این نوع صورت مناسب هستند، اما فریم‌های مستطیلی و مربعی بهترین گزینه هستند.
        """,
        'ROUND': """
            صورت گرد دارای عرض و طول تقریباً یکسان است و فاقد زوایای مشخص است.
            پیشنهاد: فریم‌های مستطیلی و مربعی که باعث ایجاد زاویه می‌شوند، مناسب هستند.
        """,
        'SQUARE': """
            صورت مربعی دارای فک زاویه‌دار و پهنای پیشانی و فک تقریباً یکسان است.
            پیشنهاد: فریم‌های گرد و بیضی که خطوط صورت را نرم‌تر می‌کنند، مناسب هستند.
        """,
        'HEART': """
            صورت قلبی دارای پیشانی پهن و فک باریک است.
            پیشنهاد: فریم‌های گرد و بیضی که در قسمت پایین پهن‌تر هستند، مناسب هستند.
        """,
        'OBLONG': """
            صورت کشیده دارای طول بیشتر نسبت به عرض است.
            پیشنهاد: فریم‌های گرد و مربعی با عمق بیشتر مناسب هستند.
        """,
        'DIAMOND': """
            صورت لوزی دارای گونه‌های برجسته و پیشانی و فک باریک است.
            پیشنهاد: فریم‌های گربه‌ای و بیضی که خط ابرو را برجسته می‌کنند، مناسب هستند.
        """,
        'TRIANGLE': """
            صورت مثلثی دارای پیشانی باریک و فک پهن است.
            پیشنهاد: فریم‌های که در قسمت بالا پررنگ‌تر هستند، مناسب هستند.
        """
    }
    
    return descriptions.get(face_shape, "توضیحات این شکل صورت در دسترس نیست.")


def get_frame_recommendations(face_shape: str) -> List[str]:
    """
    Get recommended frame types for a face shape.
    
    Args:
        face_shape: Face shape (OVAL, ROUND, etc.)
        
    Returns:
        list: Recommended frame types
    """
    recommendations = {
        'OVAL': ['مستطیلی', 'مربعی', 'هشت‌ضلعی', 'گربه‌ای', 'بیضی'],  # Can wear most frames
        'ROUND': ['مستطیلی', 'مربعی', 'هشت‌ضلعی', 'هاوایی'],  # Angular frames add definition
        'SQUARE': ['گرد', 'بیضی', 'گربه‌ای', 'هاوایی'],  # Curved frames soften angles
        'HEART': ['گرد', 'بیضی', 'هاوایی', 'پایین‌بدون‌فریم'],  # Frames wider at bottom
        'OBLONG': ['مربعی', 'گرد', 'گربه‌ای', 'هاوایی'],  # Frames with depth
        'DIAMOND': ['گربه‌ای', 'هاوایی', 'بیضی', 'بدون‌فریم'],  # Frames highlighting brow line
        'TRIANGLE': ['گربه‌ای', 'مستطیلی', 'هاوایی', 'بالا‌پررنگ']  # Frames emphasizing top part
    }
    
    return recommendations.get(face_shape, ['مستطیلی', 'گرد'])  # Default recommendation