import logging
import aiohttp
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_all_products() -> List[Dict[str, Any]]:
    """
    Fetch all products from WooCommerce API.
    
    Returns:
        list: Products from WooCommerce API
    """
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
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Make API request
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "per_page": per_page,
                    "page": page
                }
                
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                        break
                        
                    products = await response.json()
                    
                    if not products:
                        break
                        
                    all_products.extend(products)
                    
                    # Check if we've received less than the maximum per page
                    if len(products) < per_page:
                        break
                        
                    page += 1
        
        logger.info(f"Fetched {len(all_products)} products from WooCommerce API")
        return all_products
        
    except Exception as e:
        logger.error(f"Error fetching WooCommerce products: {str(e)}")
        return []


async def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific product by ID from WooCommerce API.
    
    Args:
        product_id: Product ID
        
    Returns:
        dict: Product data or None if not found
    """
    try:
        logger.info(f"Fetching product {product_id} from WooCommerce API")
        
        # WooCommerce API credentials
        api_url = f"{settings.woocommerce_api_url}/{product_id}"
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        
        params = {
            "consumer_key": consumer_key,
            "consumer_secret": consumer_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                    return None
                    
                product = await response.json()
                return product
                
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {str(e)}")
        return None


async def get_products_by_category(category_id: int) -> List[Dict[str, Any]]:
    """
    Fetch products by category ID from WooCommerce API.
    
    Args:
        category_id: Category ID
        
    Returns:
        list: Products in the category
    """
    try:
        logger.info(f"Fetching products in category {category_id} from WooCommerce API")
        
        # WooCommerce API credentials
        api_url = settings.woocommerce_api_url
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        per_page = settings.woocommerce_per_page
        
        # Initialize products list
        category_products = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Make API request
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "category": category_id,
                    "per_page": per_page,
                    "page": page
                }
                
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                        break
                        
                    products = await response.json()
                    
                    if not products:
                        break
                        
                    category_products.extend(products)
                    
                    # Check if we've received less than the maximum per page
                    if len(products) < per_page:
                        break
                        
                    page += 1
        
        logger.info(f"Fetched {len(category_products)} products in category {category_id}")
        return category_products
        
    except Exception as e:
        logger.error(f"Error fetching products in category {category_id}: {str(e)}")
        return []


async def get_products_by_attribute(attribute_id: int, term_id: int) -> List[Dict[str, Any]]:
    """
    Fetch products by attribute and term ID from WooCommerce API.
    
    Args:
        attribute_id: Attribute ID
        term_id: Term ID
        
    Returns:
        list: Products with the attribute term
    """
    try:
        logger.info(f"Fetching products with attribute {attribute_id} and term {term_id}")
        
        # WooCommerce API credentials
        api_url = settings.woocommerce_api_url
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        per_page = settings.woocommerce_per_page
        
        # Initialize products list
        attribute_products = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Make API request
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "attribute": attribute_id,
                    "attribute_term": term_id,
                    "per_page": per_page,
                    "page": page
                }
                
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                        break
                        
                    products = await response.json()
                    
                    if not products:
                        break
                        
                    attribute_products.extend(products)
                    
                    # Check if we've received less than the maximum per page
                    if len(products) < per_page:
                        break
                        
                    page += 1
        
        logger.info(f"Fetched {len(attribute_products)} products with attribute {attribute_id} and term {term_id}")
        return attribute_products
        
    except Exception as e:
        logger.error(f"Error fetching products with attribute {attribute_id} and term {term_id}: {str(e)}")
        return []


async def get_product_categories() -> List[Dict[str, Any]]:
    """
    Fetch product categories from WooCommerce API.
    
    Returns:
        list: Product categories
    """
    try:
        logger.info("Fetching product categories from WooCommerce API")
        
        # WooCommerce API credentials
        api_url = f"{settings.woocommerce_api_url}/categories"
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        per_page = 100
        
        # Initialize categories list
        all_categories = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Make API request
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "per_page": per_page,
                    "page": page
                }
                
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                        break
                        
                    categories = await response.json()
                    
                    if not categories:
                        break
                        
                    all_categories.extend(categories)
                    
                    # Check if we've received less than the maximum per page
                    if len(categories) < per_page:
                        break
                        
                    page += 1
        
        logger.info(f"Fetched {len(all_categories)} product categories")
        return all_categories
        
    except Exception as e:
        logger.error(f"Error fetching product categories: {str(e)}")
        return []


async def get_product_attributes() -> List[Dict[str, Any]]:
    """
    Fetch product attributes from WooCommerce API.
    
    Returns:
        list: Product attributes
    """
    try:
        logger.info("Fetching product attributes from WooCommerce API")
        
        # WooCommerce API credentials
        api_url = f"{settings.woocommerce_api_url}/attributes"
        consumer_key = settings.woocommerce_consumer_key
        consumer_secret = settings.woocommerce_consumer_secret
        
        params = {
            "consumer_key": consumer_key,
            "consumer_secret": consumer_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"WooCommerce API error: {response.status} - {error_text}")
                    return []
                    
                attributes = await response.json()
                logger.info(f"Fetched {len(attributes)} product attributes")
                return attributes
                
    except Exception as e:
        logger.error(f"Error fetching product attributes: {str(e)}")
        return []