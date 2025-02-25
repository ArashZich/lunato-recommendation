import base64
import re
import logging
from typing import Optional, Union
import io
from PIL import Image

logger = logging.getLogger(__name__)

# Regular expression for matching base64 image data
BASE64_PATTERN = re.compile(r'^data:image/([a-zA-Z]+);base64,([^\"]*)|^([a-zA-Z0-9+/=]+)$')

# Maximum image size in bytes (10MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Allowed image formats
ALLOWED_IMAGE_FORMATS = ['jpeg', 'jpg', 'png']


def validate_image(image_data: str) -> bool:
    """
    Validate that the provided data is a valid base64 encoded image.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if image data is not empty
        if not image_data:
            logger.warning("Empty image data")
            return False
        
        # Extract base64 data
        match = BASE64_PATTERN.match(image_data)
        if not match:
            logger.warning("Image data doesn't match base64 pattern")
            return False
            
        # If it's a data URL, extract the content part
        if match.group(1) and match.group(2):
            image_format = match.group(1).lower()
            base64_data = match.group(2)
        else:
            # If it's just base64 without data URL format, assume it's JPEG
            image_format = 'jpeg'
            base64_data = match.group(3) or image_data
            
        # Check if image format is allowed
        if image_format not in ALLOWED_IMAGE_FORMATS:
            logger.warning(f"Image format {image_format} not allowed")
            return False
            
        # Check if base64 data is valid
        try:
            # Try to decode base64
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            logger.warning(f"Invalid base64 encoding: {str(e)}")
            return False
            
        # Check image size
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large: {len(image_bytes)} bytes")
            return False
            
        # Try to open the image to verify it's valid
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
            return True
        except Exception as e:
            logger.warning(f"Invalid image data: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False


def validate_face_shape(face_shape: str) -> bool:
    """
    Validate that the provided face shape is one of the allowed values.
    
    Args:
        face_shape: Face shape string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # List of valid face shapes
    valid_shapes = ['OVAL', 'ROUND', 'SQUARE', 'HEART', 'OBLONG', 'DIAMOND', 'TRIANGLE']
    
    return face_shape.upper() in valid_shapes


def extract_base64_data(image_data: str) -> Optional[bytes]:
    """
    Extract binary data from base64 string.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        bytes: Decoded binary data or None if invalid
    """
    try:
        # Extract base64 data
        match = BASE64_PATTERN.match(image_data)
        if not match:
            return None
            
        # If it's a data URL, extract the content part
        if match.group(1) and match.group(2):
            base64_data = match.group(2)
        else:
            # If it's just base64 without data URL format
            base64_data = match.group(3) or image_data
            
        # Decode base64
        return base64.b64decode(base64_data)
        
    except Exception as e:
        logger.error(f"Error extracting base64 data: {str(e)}")
        return None