# Import utility functions to make them available via the api.utils namespace
from api.utils.client_info import extract_client_info, get_device_category, get_browser_family
from api.utils.validators import validate_image, validate_face_shape, extract_base64_data
from api.utils.image_processing import (
    base64_to_opencv, opencv_to_base64, resize_image, 
    enhance_image, crop_face, normalize_image
)