import base64
import cv2
import numpy as np
import io
from PIL import Image
import logging
from typing import Tuple, Optional, Dict, Any, List
from api.utils.validators import extract_base64_data

logger = logging.getLogger(__name__)


def base64_to_opencv(image_data: str) -> Optional[np.ndarray]:
    """
    Convert base64 encoded image to OpenCV format.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        numpy.ndarray: OpenCV image or None if conversion fails
    """
    try:
        # Extract binary data from base64
        image_bytes = extract_base64_data(image_data)
        
        if image_bytes is None:
            return None
            
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Error converting base64 to OpenCV image: {str(e)}")
        return None


def opencv_to_base64(image: np.ndarray, format: str = 'jpeg') -> Optional[str]:
    """
    Convert OpenCV image to base64 encoded string.
    
    Args:
        image: OpenCV image (numpy array)
        format: Image format ('jpeg', 'png')
        
    Returns:
        str: Base64 encoded image or None if conversion fails
    """
    try:
        # Encode image
        if format.lower() == 'png':
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
        # Convert to base64
        base64_data = base64.b64encode(buffer).decode('utf-8')
        
        # Return as data URL
        return f"data:image/{format.lower()};base64,{base64_data}"
        
    except Exception as e:
        logger.error(f"Error converting OpenCV image to base64: {str(e)}")
        return None


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image if larger than max_size while maintaining aspect ratio.
    
    Args:
        image: OpenCV image
        max_size: Maximum dimension (width or height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # If image is smaller than max_size, return as is
    if width <= max_size and height <= max_size:
        return image
        
    # Calculate scaling factor
    scaling_factor = max_size / max(width, height)
    
    # Calculate new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better face detection.
    
    Args:
        image: OpenCV image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to color
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def crop_face(image: np.ndarray, face_coordinates: Dict[str, int], margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with margin.
    
    Args:
        image: OpenCV image
        face_coordinates: Dictionary with x, y, width, height
        margin: Margin around face as percentage of face size
        
    Returns:
        numpy.ndarray: Cropped face image
    """
    # Extract face coordinates
    x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['width'], face_coordinates['height']
    
    # Calculate margin in pixels
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    # Calculate crop coordinates with margin
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)
    
    # Crop image
    return image[y1:y2, x1:x2]


def draw_face_landmarks(image: np.ndarray, landmarks: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0), radius: int = 2) -> np.ndarray:
    """
    Draw facial landmarks on image.
    
    Args:
        image: OpenCV image
        landmarks: List of (x, y) coordinates
        color: BGR color tuple
        radius: Point radius
        
    Returns:
        numpy.ndarray: Image with landmarks
    """
    # Create copy of image
    result = image.copy()
    
    # Draw each landmark point
    for point in landmarks:
        cv2.circle(result, point, radius, color, -1)
        
    return result


def annotate_face_shape(image: np.ndarray, face_shape: str, face_coordinates: Dict[str, int]) -> np.ndarray:
    """
    Annotate image with face shape text.
    
    Args:
        image: OpenCV image
        face_shape: Face shape string
        face_coordinates: Dictionary with x, y, width, height
        
    Returns:
        numpy.ndarray: Annotated image
    """
    # Create copy of image
    result = image.copy()
    
    # Extract face coordinates
    x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['width'], face_coordinates['height']
    
    # Draw rectangle around face
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw face shape text
    text = f"Face Shape: {face_shape}"
    cv2.putText(result, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for ML processing.
    
    Args:
        image: OpenCV image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Normalize pixel values to [0, 1]
    return gray.astype(np.float32) / 255.0