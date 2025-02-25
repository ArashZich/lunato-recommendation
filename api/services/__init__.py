# Import services to make them available via the api.services namespace
from api.services.face_detection import detect_face_local, get_face_region
from api.services.face_analysis import analyze_face_shape, get_frame_recommendations
from api.services.frame_matcher import match_frames
from api.services.woocommerce import get_all_products, get_product_by_id
from api.services.analytics import generate_analytics, track_user_activity