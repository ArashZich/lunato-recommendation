import numpy as np
import logging
import pickle
import os
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

logger = logging.getLogger(__name__)

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models", "face_shape_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "trained_models", "face_shape_scaler.pkl")

# Facial shape categories
FACE_SHAPES = ['OVAL', 'ROUND', 'SQUARE', 'HEART', 'OBLONG', 'DIAMOND', 'TRIANGLE']


class FaceShapeClassifier:
    """
    Classifier for determining face shape from facial features.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """
        Initialize the face shape classifier.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Try to load pre-trained model
        self._load_model()
        
    def _load_model(self):
        """
        Load the pre-trained model and scaler.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Loaded pre-trained face shape model")
            else:
                logger.warning(f"Pre-trained model not found at {self.model_path}")
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded feature scaler")
            else:
                logger.warning(f"Feature scaler not found at {self.scaler_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def train(self, features: List[Dict[str, float]], labels: List[str]):
        """
        Train the classifier on facial features.
        
        Args:
            features: List of feature dictionaries
            labels: List of face shape labels
        """
        try:
            # Extract feature values as a list of lists
            X = []
            for feature_dict in features:
                X.append([
                    feature_dict.get('width_to_length_ratio', 0),
                    feature_dict.get('cheekbone_to_jaw_ratio', 0),
                    feature_dict.get('forehead_to_cheekbone_ratio', 0),
                    feature_dict.get('jaw_angle', 0)
                ])
            
            X = np.array(X)
            y = np.array(labels)
            
            # Create and fit the scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the SVM classifier
            self.model = SVC(kernel='rbf', probability=True)
            self.model.fit(X_scaled, y)
            
            # Save the model and scaler
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info("Face shape classifier trained and saved")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict face shape from features.
        
        Args:
            features: Dictionary of facial features
            
        Returns:
            tuple: (face_shape, confidence, shape_probabilities)
        """
        # If model is not loaded, fall back to rule-based classification
        if self.model is None or self.scaler is None:
            return self._rule_based_classification(features)
        
        try:
            # Extract feature values
            X = np.array([[
                features.get('width_to_length_ratio', 0),
                features.get('cheekbone_to_jaw_ratio', 0),
                features.get('forehead_to_cheekbone_ratio', 0),
                features.get('jaw_angle', 0)
            ]])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predicted probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get shape with highest probability
            prediction_idx = np.argmax(probabilities)
            face_shape = self.model.classes_[prediction_idx]
            confidence = probabilities[prediction_idx] * 100
            
            # Create probability dictionary
            probability_dict = {}
            for i, shape in enumerate(self.model.classes_):
                probability_dict[shape] = float(probabilities[i])
            
            return face_shape, confidence, probability_dict
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return self._rule_based_classification(features)
    
    def _rule_based_classification(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Rule-based classification as backup when model is not available.
        
        Args:
            features: Dictionary of facial features
            
        Returns:
            tuple: (face_shape, confidence, shape_scores)
        """
        # Extract features
        width_to_length = features.get('width_to_length_ratio', 0)
        cheekbone_to_jaw = features.get('cheekbone_to_jaw_ratio', 0)
        forehead_to_cheekbone = features.get('forehead_to_cheekbone_ratio', 0)
        jaw_angle = features.get('jaw_angle', 0)
        
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
        if 0.7 <= width_to_length <= 0.8:
            shape_scores['OVAL'] += 3
        if 1.0 <= cheekbone_to_jaw <= 1.1:
            shape_scores['OVAL'] += 2
        if 120 <= jaw_angle <= 140:
            shape_scores['OVAL'] += 1
        
        # Round face shape (width similar to length)
        if 0.9 <= width_to_length <= 1.1:
            shape_scores['ROUND'] += 3
        if 1.1 <= cheekbone_to_jaw <= 1.3:
            shape_scores['ROUND'] += 2
        if jaw_angle >= 130:
            shape_scores['ROUND'] += 1
        
        # Square face shape (angular jaw, width similar to length)
        if 0.9 <= width_to_length <= 1.0:
            shape_scores['SQUARE'] += 2
        if 0.9 <= cheekbone_to_jaw <= 1.05:
            shape_scores['SQUARE'] += 3
        if jaw_angle <= 125:
            shape_scores['SQUARE'] += 2
        
        # Heart face shape (wider forehead, narrower jaw)
        if forehead_to_cheekbone >= 1.0:
            shape_scores['HEART'] += 3
        if cheekbone_to_jaw >= 1.3:
            shape_scores['HEART'] += 2
        
        # Oblong face shape (long face)
        if width_to_length <= 0.7:
            shape_scores['OBLONG'] += 3
        if 0.9 <= cheekbone_to_jaw <= 1.1:
            shape_scores['OBLONG'] += 1
        
        # Diamond face shape (narrow forehead and jaw, wide cheekbones)
        if forehead_to_cheekbone <= 0.9:
            shape_scores['DIAMOND'] += 2
        if cheekbone_to_jaw >= 1.2:
            shape_scores['DIAMOND'] += 2
        if jaw_angle >= 130:
            shape_scores['DIAMOND'] += 1
        
        # Triangle face shape (narrow forehead, wide jaw)
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
        
        # Convert scores to probabilities
        total_score = sum(shape_scores.values())
        if total_score > 0:
            probability_dict = {shape: score / total_score for shape, score in shape_scores.items()}
        else:
            probability_dict = {shape: 1.0 / len(shape_scores) for shape in shape_scores}
        
        return face_shape, confidence, probability_dict


# Singleton instance
classifier = FaceShapeClassifier()


def predict_face_shape(features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict face shape from features using the singleton classifier.
    
    Args:
        features: Dictionary of facial features
        
    Returns:
        tuple: (face_shape, confidence, shape_probabilities)
    """
    return classifier.predict(features)