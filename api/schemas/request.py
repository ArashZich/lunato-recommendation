from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ClientInfo(BaseModel):
    """Client device and browser information."""
    device_type: Optional[str] = Field(None, title="Device type (mobile, tablet, desktop)")
    os_name: Optional[str] = Field(None, title="Operating system name")
    os_version: Optional[str] = Field(None, title="Operating system version")
    browser_name: Optional[str] = Field(None, title="Browser name")
    browser_version: Optional[str] = Field(None, title="Browser version")
    screen_width: Optional[int] = Field(None, title="Screen width in pixels")
    screen_height: Optional[int] = Field(None, title="Screen height in pixels")
    user_agent: Optional[str] = Field(None, title="Full user agent string")
    ip_address: Optional[str] = Field(None, title="IP address")
    language: Optional[str] = Field(None, title="Browser language")
    
    class Config:
        schema_extra = {
            "example": {
                "device_type": "mobile",
                "os_name": "iOS",
                "os_version": "15.4",
                "browser_name": "Safari",
                "browser_version": "15.4",
                "screen_width": 390,
                "screen_height": 844,
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
                "ip_address": "192.168.1.1",
                "language": "fa-IR"
            }
        }


class FaceAnalysisRequest(BaseModel):
    """Request model for face analysis."""
    image: str = Field(..., title="Base64 encoded image data")
    user_id: Optional[str] = Field(None, title="User ID for tracking")
    client_info: Optional[ClientInfo] = Field(None, title="Client device information")
    
    class Config:
        schema_extra = {
            "example": {
                "image": "base64_encoded_image_data",
                "user_id": "user123",
                "client_info": {
                    "device_type": "mobile",
                    "browser_name": "Chrome"
                }
            }
        }


class FrameRecommendationRequest(BaseModel):
    """Request model for getting frame recommendations."""
    face_shape: Optional[str] = Field(None, title="Face shape (if already known)")
    detection_id: Optional[str] = Field(None, title="ID of previous face detection")
    user_id: Optional[str] = Field(None, title="User ID for tracking")
    filter_by_price: Optional[bool] = Field(False, title="Filter by price range")
    min_price: Optional[float] = Field(None, title="Minimum price")
    max_price: Optional[float] = Field(None, title="Maximum price")
    limit: Optional[int] = Field(10, title="Maximum number of recommendations to return")
    
    class Config:
        schema_extra = {
            "example": {
                "face_shape": "OVAL",
                "user_id": "user123",
                "filter_by_price": True,
                "min_price": 1000000,
                "max_price": 5000000,
                "limit": 5
            }
        }


class AnalyticsRequest(BaseModel):
    """Request model for analytics queries."""
    start_date: Optional[datetime] = Field(None, title="Start date for analytics")
    end_date: Optional[datetime] = Field(None, title="End date for analytics")
    group_by: Optional[str] = Field("face_shape", title="Group results by this field")
    
    class Config:
        schema_extra = {
            "example": {
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-12-31T23:59:59",
                "group_by": "face_shape"
            }
        }