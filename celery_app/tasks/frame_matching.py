import logging
from celery import shared_task
from pymongo import MongoClient
import requests
from api.config import get_settings
from celery_app.app import app
from datetime import datetime
import json
from datetime import timedelta

logger = logging.getLogger(__name__)
settings = get_settings()

# MongoDB connection for Celery tasks (synchronous)
mongo_client = MongoClient(settings.mongodb_uri)
db = mongo_client[settings.mongo_db_name]

# Face shape to frame type mapping
FACE_SHAPE_TO_FRAME_MAP = {
    'OVAL': ['مستطیلی', 'مربعی', 'هشت‌ضلعی', 'گربه‌ای', 'بیضی'],  # Can wear most frames
    'ROUND': ['مستطیلی', 'مربعی', 'هشت‌ضلعی', 'هاوایی'],  # Angular frames add definition
    'SQUARE': ['گرد', 'بیضی', 'گربه‌ای', 'هاوایی'],  # Curved frames soften angles
    'HEART': ['گرد', 'بیضی', 'هاوایی', 'پایین‌بدون‌فریم'],  # Frames wider at bottom
    'OBLONG': ['مربعی', 'گرد', 'گربه‌ای', 'هاوایی'],  # Frames with depth
    'DIAMOND': ['گربه‌ای', 'هاوایی', 'بیضی', 'بدون‌فریم'],  # Frames highlighting brow line
    'TRIANGLE': ['گربه‌ای', 'مستطیلی', 'هاوایی', 'بالا‌پررنگ']  # Frames emphasizing top part
}

# Cache for WooCommerce products
product_cache = None
last_cache_update = None


@shared_task(name="celery_app.tasks.frame_matching.match_frames")
def match_frames(detection_id, face_shape, user_id, metadata):
    """
    Match eyeglass frames based on face shape analysis.
    
    Args:
        detection_id (str): ID of the face detection record
        face_shape (str): Determined face shape
        user_id (str): User ID for tracking
        metadata (dict): Additional metadata
        
    Returns:
        dict: Matching results with recommended frames
    """
    try:
        logger.info(f"Starting frame matching for user {user_id} with face shape {face_shape}")
        
        # Get appropriate frame types for this face shape
        recommended_frame_types = FACE_SHAPE_TO_FRAME_MAP.get(face_shape, [])
        
        if not recommended_frame_types:
            logger.warning(f"No frame types mapped for face shape {face_shape}")
            return {
                "success": False,
                "message": f"No frame recommendations available for face shape {face_shape}"
            }
        
        # Get products from WooCommerce API
        products = get_woocommerce_products()
        
        if not products:
            logger.error("Failed to fetch products from WooCommerce API")
            return {
                "success": False,
                "message": "Failed to fetch available frames"
            }
        
        # Filter eyeglass frames by shape
        matched_frames = []
        
        # Extract frames with the recommended shapes
        for product in products:
            # Skip non-eyeglass products or products without necessary attributes
            if not is_eyeglass_frame(product):
                continue
                
            product_frame_type = get_frame_type(product)
            
            if product_frame_type in recommended_frame_types:
                matched_frames.append({
                    "id": product["id"],
                    "name": product["name"],
                    "permalink": product["permalink"],
                    "price": product.get("price", ""),
                    "regular_price": product.get("regular_price", ""),
                    "frame_type": product_frame_type,
                    "images": [img["src"] for img in product.get("images", [])[:3]],
                    "match_score": calculate_match_score(face_shape, product_frame_type)
                })
        
        # Sort by match score (highest first)
        matched_frames.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Limit to top 10 recommendations
        top_recommendations = matched_frames[:10]
        
        # Save recommendations to database
        recommendation_record = {
            "user_id": user_id,
            "detection_id": detection_id,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommendations": top_recommendations,
            "metadata": metadata,
            "created_at": datetime.utcnow()
        }
        
        db.recommendations.insert_one(recommendation_record)
        
        # Update analytics
        update_analytics(user_id, face_shape, top_recommendations)
        
        logger.info(f"Frame matching completed for user {user_id}: {len(top_recommendations)} recommendations")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommendations": top_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error in frame matching: {str(e)}")
        return {
            "success": False,
            "message": f"Error matching frames: {str(e)}"
        }


def get_woocommerce_products():
    """
    Fetch products from WooCommerce API with caching
    
    Returns:
        list: WooCommerce products
    """
    global product_cache, last_cache_update
    
    # Use cache if available and less than 1 hour old
    if product_cache and last_cache_update and (datetime.utcnow() - last_cache_update < timedelta(hours=1)):
        logger.info("Using cached WooCommerce products")
        return product_cache
    
    try:
        logger.info("Fetching products from WooCommerce API")
        
        # WooCommerce API credentials
        api_url = settings.woocommerce_api_url
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        per_page = settings.woocommerce_per_page
        
        # Initialize products list
        all_products = []
        page = 1
        
        while True:
            # Make API request
            response = requests.get(
                api_url,
                params={
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "per_page": per_page,
                    "page": page
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"WooCommerce API error: {response.status_code} - {response.text}")
                break
                
            products = response.json()
            
            if not products:
                break
                
            all_products.extend(products)
            
            # Check if we've received less than the maximum per page
            if len(products) < per_page:
                break
                
            page += 1
            
        # Update cache
        product_cache = all_products
        last_cache_update = datetime.utcnow()
        
        logger.info(f"Fetched {len(all_products)} products from WooCommerce API")
        return all_products
        
    except Exception as e:
        logger.error(f"Error fetching WooCommerce products: {str(e)}")
        return []


def is_eyeglass_frame(product):
    """
    Check if a product is an eyeglass frame
    
    Args:
        product: WooCommerce product
        
    Returns:
        bool: True if the product is an eyeglass frame
    """
    # Check product categories
    categories = product.get("categories", [])
    for category in categories:
        category_name = category.get("name", "").lower()
        if "عینک" in category_name or "frame" in category_name or "eyeglass" in category_name:
            return True
    
    # Check product attributes for frame related attributes
    attributes = product.get("attributes", [])
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        if "frame" in attr_name or "شکل" in attr_name or "فریم" in attr_name or "نوع" in attr_name:
            return True
    
    # Check product name for eyeglass related keywords
    name = product.get("name", "").lower()
    eyeglass_keywords = ["عینک", "فریم", "eyeglass", "glasses", "frame"]
    for keyword in eyeglass_keywords:
        if keyword in name:
            return True
    
    return False


def get_frame_type(product):
    """
    Extract the frame type from a product
    
    Args:
        product: WooCommerce product
        
    Returns:
        str: Frame type or None if not found
    """
    # Look for frame type in attributes
    attributes = product.get("attributes", [])
    
    frame_type_attrs = ["شکل فریم", "نوع فریم", "فرم فریم", "frame type", "frame shape"]
    
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        
        # Check if this attribute is related to frame type
        is_frame_type_attr = any(frame_type in attr_name for frame_type in frame_type_attrs)
        
        if is_frame_type_attr:
            # Get the value of the attribute
            if "options" in attribute and attribute["options"]:
                # Return the first option
                return attribute["options"][0]
    
    # If no specific frame type found, try to infer from the product name
    name = product.get("name", "").lower()
    
    frame_types_mapping = {
        "مستطیلی": ["مستطیل", "rectangular", "rectangle"],
        "مربعی": ["مربع", "square"],
        "گرد": ["گرد", "round", "circular"],
        "بیضی": ["بیضی", "oval"],
        "گربه‌ای": ["گربه", "cat eye", "cat-eye"],
        "هشت‌ضلعی": ["هشت", "octagonal", "octagon"],
        "هاوایی": ["هاوایی", "aviator"],
        "بدون‌فریم": ["بدون فریم", "rimless"]
    }
    
    for frame_type, keywords in frame_types_mapping.items():
        for keyword in keywords:
            if keyword in name:
                return frame_type
    
    # Default to a common type if can't determine
    return "مستطیلی"


def calculate_match_score(face_shape, frame_type):
    """
    Calculate a match score between face shape and frame type
    
    Args:
        face_shape: Face shape
        frame_type: Frame type
        
    Returns:
        float: Match score (0-100)
    """
    # Get recommended frame types for this face shape
    recommended_types = FACE_SHAPE_TO_FRAME_MAP.get(face_shape, [])
    
    if not recommended_types:
        return 0
    
    # If the frame type is in the top 2 recommended types, give it a high score
    if frame_type in recommended_types[:2]:
        return 90 + (recommended_types.index(frame_type) * -5)
    
    # If it's in the recommended list but not in top 2, medium score
    if frame_type in recommended_types:
        position = recommended_types.index(frame_type)
        return 80 - (position * 5)
    
    # If it's not in the recommended list, low score
    return 40


def update_analytics(user_id, face_shape, recommendations):
    """
    Update analytics data for recommendation insights
    
    Args:
        user_id: User ID
        face_shape: Face shape
        recommendations: List of recommended products
    """
    try:
        analytics_record = {
            "user_id": user_id,
            "face_shape": face_shape,
            "recommendations_count": len(recommendations),
            "recommended_product_ids": [rec["id"] for rec in recommendations],
            "timestamp": datetime.utcnow()
        }
        
        db.recommendation_analytics.insert_one(analytics_record)
        
    except Exception as e:
        logger.error(f"Error updating analytics: {str(e)}")