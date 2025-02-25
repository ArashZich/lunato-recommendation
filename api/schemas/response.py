from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = Field(..., title="Success status")
    message: Optional[str] = Field(None, title="Response message")


class FaceCoordinates(BaseModel):
    """Face coordinates in the image."""
    x: int = Field(..., title="X coordinate of top-left corner")
    y: int = Field(..., title="Y coordinate of top-left corner")
    width: int = Field(..., title="Width of face")
    height: int = Field(..., title="Height of face")
    center_x: int = Field(..., title="X coordinate of face center")
    center_y: int = Field(..., title="Y coordinate of face center")
    aspect_ratio: float = Field(..., title="Aspect ratio (width/height)")


class FaceDetectionResponse(BaseResponse):
    """Response model for face detection."""
    detection_id: Optional[str] = Field(None, title="Detection ID for future reference")
    face: Optional[FaceCoordinates] = Field(None, title="Detected face coordinates")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Face detected successfully",
                "detection_id": "550e8400-e29b-41d4-a716-446655440000",
                "face": {
                    "x": 100,
                    "y": 80,
                    "width": 200,
                    "height": 220,
                    "center_x": 200,
                    "center_y": 190,
                    "aspect_ratio": 0.91
                }
            }
        }


class FaceAnalysisResponse(BaseResponse):
    """Response model for face shape analysis."""
    face_shape: Optional[str] = Field(None, title="Determined face shape")
    confidence: Optional[float] = Field(None, title="Confidence percentage")
    shape_metrics: Optional[Dict[str, float]] = Field(None, title="Scores for each face shape")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Face shape analysis completed",
                "face_shape": "OVAL",
                "confidence": 85.5,
                "shape_metrics": {
                    "OVAL": 5,
                    "ROUND": 3,
                    "SQUARE": 1,
                    "HEART": 0,
                    "OBLONG": 1,
                    "DIAMOND": 0,
                    "TRIANGLE": 0
                }
            }
        }


class FrameImage(BaseModel):
    """Eyeglass frame image."""
    src: str = Field(..., title="Image URL")


class RecommendedFrame(BaseModel):
    """Recommended eyeglass frame."""
    id: int = Field(..., title="Product ID")
    name: str = Field(..., title="Product name")
    permalink: str = Field(..., title="Product URL")
    price: str = Field(..., title="Product price")
    regular_price: Optional[str] = Field(None, title="Regular price before discount")
    frame_type: str = Field(..., title="Frame type/shape")
    images: List[str] = Field(..., title="Product images")
    match_score: float = Field(..., title="Match score for this face shape")


class FrameRecommendationResponse(BaseResponse):
    """Response model for frame recommendations."""
    face_shape: Optional[str] = Field(None, title="Face shape used for recommendations")
    recommended_frame_types: Optional[List[str]] = Field(None, title="Recommended frame types for this face shape")
    recommendations: Optional[List[RecommendedFrame]] = Field(None, title="Recommended frames")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Recommendations generated successfully",
                "face_shape": "OVAL",
                "recommended_frame_types": ["مستطیلی", "مربعی", "هشت‌ضلعی"],
                "recommendations": [
                    {
                        "id": 123,
                        "name": "فریم طبی مدل مستطیلی کلاسیک",
                        "permalink": "https://lunato.shop/product/classic-rectangular",
                        "price": "2500000",
                        "regular_price": "3000000",
                        "frame_type": "مستطیلی",
                        "images": ["https://lunato.shop/wp-content/uploads/frame1.jpg"],
                        "match_score": 92.5
                    }
                ]
            }
        }


class FaceShapeCount(BaseModel):
    """Face shape count for analytics."""
    face_shape: str = Field(..., title="Face shape")
    count: int = Field(..., title="Count")
    percentage: float = Field(..., title="Percentage of total")


class FrameTypeCount(BaseModel):
    """Frame type recommendation count for analytics."""
    frame_type: str = Field(..., title="Frame type")
    count: int = Field(..., title="Count")
    percentage: float = Field(..., title="Percentage of total")


class DeviceAnalytics(BaseModel):
    """Device analytics."""
    device_type: str = Field(..., title="Device type")
    count: int = Field(..., title="Count")
    percentage: float = Field(..., title="Percentage of total")


class BrowserAnalytics(BaseModel):
    """Browser analytics."""
    browser_name: str = Field(..., title="Browser name")
    count: int = Field(..., title="Count")
    percentage: float = Field(..., title="Percentage of total")


class AnalyticsResponse(BaseResponse):
    """Response model for analytics."""
    total_analyses: int = Field(..., title="Total number of face analyses")
    face_shapes: List[FaceShapeCount] = Field(..., title="Face shape distribution")
    recommended_frames: List[FrameTypeCount] = Field(..., title="Recommended frame types distribution")
    devices: List[DeviceAnalytics] = Field(..., title="Device distribution")
    browsers: List[BrowserAnalytics] = Field(..., title="Browser distribution")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Analytics generated successfully",
                "total_analyses": 1250,
                "face_shapes": [
                    {"face_shape": "OVAL", "count": 450, "percentage": 36.0},
                    {"face_shape": "ROUND", "count": 320, "percentage": 25.6}
                ],
                "recommended_frames": [
                    {"frame_type": "مستطیلی", "count": 580, "percentage": 46.4},
                    {"frame_type": "گرد", "count": 350, "percentage": 28.0}
                ],
                "devices": [
                    {"device_type": "mobile", "count": 820, "percentage": 65.6},
                    {"device_type": "desktop", "count": 430, "percentage": 34.4}
                ],
                "browsers": [
                    {"browser_name": "Chrome", "count": 650, "percentage": 52.0},
                    {"browser_name": "Safari", "count": 420, "percentage": 33.6}
                ]
            }
        }