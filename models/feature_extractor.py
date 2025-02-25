import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)


class FacialFeatureExtractor:
    """
    Extract facial features from images and facial landmarks.
    """
    
    def __init__(self):
        """
        Initialize the feature extractor.
        """
        pass
    
    def extract_features_from_landmarks(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract face shape features from facial landmarks.
        
        Args:
            landmarks: Numpy array of landmark coordinates (68 landmarks)
            
        Returns:
            dict: Features extracted from landmarks
        """
        try:
            # Ensure landmarks is a numpy array
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)
            
            # Calculate key measurements using facial landmarks
            # Note: These indices correspond to the 68-point facial landmark model
            
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
            width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
            cheekbone_to_jaw_ratio = cheekbone_width / jawline_width if jawline_width > 0 else 0
            forehead_to_cheekbone_ratio = forehead_width / cheekbone_width if cheekbone_width > 0 else 0
            
            # Jaw angle (in degrees)
            chin_to_jaw_left_vec = landmarks[3] - landmarks[8]
            chin_to_jaw_right_vec = landmarks[13] - landmarks[8]
            
            cos_angle = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec) / (
                np.linalg.norm(chin_to_jaw_left_vec) * np.linalg.norm(chin_to_jaw_right_vec)
            )
            jaw_angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
            
            # Return all features
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
            
        except Exception as e:
            logger.error(f"Error extracting features from landmarks: {str(e)}")
            return {
                'width_to_length_ratio': 0.75,  # Default to oval face
                'cheekbone_to_jaw_ratio': 1.0,
                'forehead_to_cheekbone_ratio': 1.0,
                'jaw_angle': 130.0,
                'forehead_width': 100.0,
                'cheekbone_width': 100.0,
                'jawline_width': 100.0,
                'face_length': 130.0,
                'jawline_length': 100.0,
            }
    
    def extract_features_from_image(self, face_image: np.ndarray, face_detector, landmark_predictor) -> Optional[Dict[str, float]]:
        """
        Extract face shape features from a face image.
        
        Args:
            face_image: OpenCV face image
            face_detector: dlib face detector
            landmark_predictor: dlib landmark predictor
            
        Returns:
            dict: Features extracted from image or None if face not detected
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = face_detector(gray, 1)
            
            if not faces:
                logger.warning("No face detected in the image")
                return None
            
            # Get the first face
            face = faces[0]
            
            # Predict landmarks
            landmarks = landmark_predictor(gray, face)
            
            # Convert to numpy array
            points = []
            for i in range(68):
                points.append((landmarks.part(i).x, landmarks.part(i).y))
            
            landmarks_np = np.array(points)
            
            # Extract features from landmarks
            return self.extract_features_from_landmarks(landmarks_np)
            
        except Exception as e:
            logger.error(f"Error extracting features from image: {str(e)}")
            return None
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize feature values to appropriate ranges.
        
        Args:
            features: Dictionary of raw features
            
        Returns:
            dict: Normalized features
        """
        # Define expected ranges for key ratios
        ratio_ranges = {
            'width_to_length_ratio': (0.6, 1.2),
            'cheekbone_to_jaw_ratio': (0.8, 1.5),
            'forehead_to_cheekbone_ratio': (0.8, 1.2),
            'jaw_angle': (90, 150)
        }
        
        # Clone features
        normalized = features.copy()
        
        # Clip values to expected ranges
        for feature, (min_val, max_val) in ratio_ranges.items():
            if feature in normalized:
                normalized[feature] = max(min_val, min(normalized[feature], max_val))
        
        return normalized


# Create singleton instance
feature_extractor = FacialFeatureExtractor()


def extract_face_features(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Extract face features using the singleton extractor.
    
    Args:
        landmarks: Numpy array of landmark coordinates
        
    Returns:
        dict: Extracted features
    """
    return feature_extractor.extract_features_from_landmarks(landmarks)


def normalize_face_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize face features using the singleton extractor.
    
    Args:
        features: Dictionary of raw features
        
    Returns:
        dict: Normalized features
    """
    return feature_extractor.normalize_features(features)