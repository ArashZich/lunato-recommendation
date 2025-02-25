# Import model components to make them available via the models namespace
from models.face_shape_classifier import FaceShapeClassifier, predict_face_shape
from models.feature_extractor import extract_face_features, normalize_face_features