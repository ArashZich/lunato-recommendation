import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

from api.services.woocommerce import get_all_products
from api.services.face_analysis import get_frame_recommendations
from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache for products
product_cache = None
last_cache_update = None


async def match_frames(face_shape: str, user_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Match eyeglass frames based on face shape.
    
    Args:
        face_shape: Determined face shape
        user_id: User ID for tracking (optional)
        filters: Additional filters such as price range
        
    Returns:
        dict: Matching results with recommended frames
    """
    try:
        logger.info(f"Starting frame matching for face shape: {face_shape}")
        
        # Get appropriate frame types for this face shape
        recommended_frame_types = get_frame_recommendations(face_shape)
        
        if not recommended_frame_types:
            logger.warning(f"No frame types recommended for face shape {face_shape}")
            return {
                "success": False,
                "message": f"No frame recommendations available for face shape {face_shape}"
            }
        
        # Get products from WooCommerce API
        products = await get_woocommerce_products()
        
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
                # Calculate match score
                match_score = calculate_match_score(face_shape, product_frame_type, recommended_frame_types)
                
                # Apply additional filters
                if filters and not passes_filters(product, filters):
                    continue
                
                # Add to matched frames
                matched_frames.append({
                    "id": product["id"],
                    "name": product["name"],
                    "permalink": product["permalink"],
                    "price": product.get("price", ""),
                    "regular_price": product.get("regular_price", ""),
                    "frame_type": product_frame_type,
                    "images": [img["src"] for img in product.get("images", [])[:3]],
                    "match_score": match_score
                })
        
        # Sort by match score (highest first)
        matched_frames.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Limit to top recommendations (default 10)
        limit = filters.get("limit", 10) if filters else 10
        top_recommendations = matched_frames[:limit]
        
        logger.info(f"Frame matching completed: {len(top_recommendations)} recommendations found")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommendations": top_recommendations,
            "total_matches": len(matched_frames)
        }
        
    except Exception as e:
        logger.error(f"Error in frame matching: {str(e)}")
        return {
            "success": False,
            "message": f"Error matching frames: {str(e)}"
        }


async def get_woocommerce_products() -> List[Dict[str, Any]]:
    """
    Get products from WooCommerce with caching.
    
    Returns:
        list: Products from WooCommerce
    """
    global product_cache, last_cache_update
    
    # Use cache if available and less than 1 hour old
    cache_valid = (
        product_cache is not None and 
        last_cache_update is not None and 
        datetime.utcnow() - last_cache_update < timedelta(hours=1)
    )
    
    if cache_valid:
        logger.info("Using cached WooCommerce products")
        return product_cache
    
    # Fetch products from WooCommerce API
    products = await get_all_products()
    
    # Update cache
    product_cache = products
    last_cache_update = datetime.utcnow()
    
    logger.info(f"Fetched {len(products)} products from WooCommerce API")
    return products


def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
    """
    Check if a product is an eyeglass frame.
    
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
    
    # Check product attributes
    attributes = product.get("attributes", [])
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        if "frame" in attr_name or "شکل" in attr_name or "فریم" in attr_name or "نوع" in attr_name:
            return True
    
    # Check product name
    name = product.get("name", "").lower()
    keywords = ["عینک", "فریم", "eyeglass", "glasses", "frame"]
    for keyword in keywords:
        if keyword in name:
            return True
    
    return False


def get_frame_type(product: Dict[str, Any]) -> str:
    """
    Extract the frame type from a product.
    
    Args:
        product: WooCommerce product
        
    Returns:
        str: Frame type
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


def calculate_match_score(face_shape: str, frame_type: str, recommended_types: List[str]) -> float:
    """
    Calculate match score between face shape and frame type.
    
    Args:
        face_shape: Face shape
        frame_type: Frame type
        recommended_types: List of recommended frame types
        
    Returns:
        float: Match score (0-100)
    """
    if not recommended_types:
        return 50.0  # Default middle score
    
    # If the frame type is in the top 2 recommended types, give it a high score
    if frame_type in recommended_types[:2]:
        return 90.0 + (recommended_types.index(frame_type) * -5.0)
    
    # If it's in the recommended list but not in top 2, medium score
    if frame_type in recommended_types:
        position = recommended_types.index(frame_type)
        return 80.0 - (position * 5.0)
    
    # If it's not in the recommended list, low score
    return 40.0


def passes_filters(product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if a product passes the additional filters.
    
    Args:
        product: WooCommerce product
        filters: Filters such as price range
        
    Returns:
        bool: True if product passes filters
    """
    # Price filter
    if filters.get("min_price") is not None:
        price = float(product.get("price", 0))
        if price < filters["min_price"]:
            return False
    
    if filters.get("max_price") is not None:
        price = float(product.get("price", 0))
        if price > filters["max_price"]:
            return False
    
    # Gender filter
    if filters.get("gender") is not None:
        # Check product attributes for gender
        attributes = product.get("attributes", [])
        product_gender = None
        
        for attribute in attributes:
            if attribute.get("name", "").lower() in ["gender", "جنسیت"]:
                if "options" in attribute and attribute["options"]:
                    product_gender = attribute["options"][0].lower()
                    break
        
        if product_gender and filters["gender"].lower() != product_gender:
            return False
    
    # Material filter
    if filters.get("material") is not None:
        # Check product attributes for material
        attributes = product.get("attributes", [])
        product_material = None
        
        for attribute in attributes:
            if attribute.get("name", "").lower() in ["material", "جنس", "متریال"]:
                if "options" in attribute and attribute["options"]:
                    product_material = attribute["options"][0].lower()
                    break
        
        if product_material and filters["material"].lower() not in product_material:
            return False
    
    # Brand filter
    if filters.get("brand") is not None:
        # Check product attributes for brand
        attributes = product.get("attributes", [])
        product_brand = None
        
        for attribute in attributes:
            if attribute.get("name", "").lower() in ["brand", "برند"]:
                if "options" in attribute and attribute["options"]:
                    product_brand = attribute["options"][0].lower()
                    break
        
        if product_brand and filters["brand"].lower() not in product_brand:
            return False
    
    return True