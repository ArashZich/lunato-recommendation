from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserAnalytics(BaseModel):
    """User analytics data model."""
    user_id: str = Field(..., title="User ID")
    device_type: Optional[str] = Field(None, title="Device type")
    os_name: Optional[str] = Field(None, title="Operating system")
    browser_name: Optional[str] = Field(None, title="Browser name")
    face_shape: Optional[str] = Field(None, title="Detected face shape")
    recommended_frames: Optional[List[str]] = Field(None, title="Recommended frame types")
    created_at: datetime = Field(..., title="Creation timestamp")


class AggregateAnalytics(BaseModel):
    """Aggregate analytics data model."""
    total_users: int = Field(..., title="Total number of users")
    total_analyses: int = Field(..., title="Total number of analyses")
    face_shape_distribution: Dict[str, int] = Field(..., title="Distribution of face shapes")
    device_distribution: Dict[str, int] = Field(..., title="Distribution of device types")
    browser_distribution: Dict[str, int] = Field(..., title="Distribution of browsers")
    recommendation_distribution: Dict[str, int] = Field(..., title="Distribution of recommendations")
    

class TimeSeriesData(BaseModel):
    """Time series data point."""
    date: datetime = Field(..., title="Date")
    value: int = Field(..., title="Value")


class TimeSeriesAnalytics(BaseModel):
    """Time series analytics data model."""
    analyses_over_time: List[TimeSeriesData] = Field(..., title="Number of analyses over time")
    face_shapes_over_time: Dict[str, List[TimeSeriesData]] = Field(..., title="Face shapes over time")


class ConversionAnalytics(BaseModel):
    """Conversion analytics data model."""
    total_views: int = Field(..., title="Total number of views")
    completed_analyses: int = Field(..., title="Number of completed analyses")
    viewed_recommendations: int = Field(..., title="Number of users who viewed recommendations")
    conversion_rate: float = Field(..., title="Conversion rate percentage")
    average_recommendations: float = Field(..., title="Average number of recommendations per user")


class FaceShapeInsights(BaseModel):
    """Face shape insights data model."""
    face_shape: str = Field(..., title="Face shape")
    count: int = Field(..., title="Number of users with this face shape")
    percentage: float = Field(..., title="Percentage of total users")
    most_recommended_frames: List[str] = Field(..., title="Most commonly recommended frames for this face shape")
    average_confidence: float = Field(..., title="Average confidence score for this face shape")


class FrameTypeInsights(BaseModel):
    """Frame type insights data model."""
    frame_type: str = Field(..., title="Frame type")
    count: int = Field(..., title="Number of times recommended")
    percentage: float = Field(..., title="Percentage of total recommendations")
    most_common_face_shapes: List[str] = Field(..., title="Face shapes most commonly matched with this frame")