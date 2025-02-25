import os
from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # API Settings
    api_host: str = Field(env="API_HOST", default="0.0.0.0")
    api_port: int = Field(env="API_PORT", default=8000)
    debug: bool = Field(env="DEBUG", default=False)
    environment: str = Field(env="ENVIRONMENT", default="development")
    
    # MongoDB Settings
    mongodb_uri: str = Field(env="MONGODB_URI", default="mongodb://localhost:27017")
    mongo_db_name: str = Field(env="MONGO_DB_NAME", default="eyeglass_recommendation")
    
    # Celery Settings
    celery_broker_url: str = Field(env="CELERY_BROKER_URL", default="redis://localhost:6379/0")
    celery_result_backend: str = Field(env="CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")
    
    # WooCommerce API Settings
    woocommerce_api_url: str = Field(env="WOOCOMMERCE_API_URL", default="https://lunato.shop/wp-json/wc/v3/products")
    woocommerce_consumer_key: str = Field(env="WOOCOMMERCE_CONSUMER_KEY", default="")
    woocommerce_consumer_secret: str = Field(env="WOOCOMMERCE_CONSUMER_SECRET", default="")
    woocommerce_per_page: int = Field(env="WOOCOMMERCE_PER_PAGE", default=100)
    
    # Face Analysis Settings
    face_detection_model: str = Field(
        env="FACE_DETECTION_MODEL", 
        default="haarcascade_frontalface_default.xml"
    )
    face_landmark_model: str = Field(
        env="FACE_LANDMARK_MODEL", 
        default="shape_predictor_68_face_landmarks.dat"
    )
    confidence_threshold: float = Field(env="CONFIDENCE_THRESHOLD", default=0.5)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the settings object
    """
    return Settings()