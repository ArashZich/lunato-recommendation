from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class AnalysisRequest(BaseModel):
    """
    Analysis request model for tracking requests.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    request_time: datetime = Field(default_factory=datetime.utcnow)
    client_info: Dict[str, Any] = {}
    status: str = "processing"  # processing, completed, error
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "request_time": "2023-07-15T10:30:00.000Z",
                "client_info": {
                    "device_type": "mobile",
                    "browser_name": "Chrome"
                },
                "status": "completed"
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AnalysisRequest object to dictionary.
        
        Returns:
            dict: Analysis request data as dictionary
        """
        return {
            "_id": self.id,
            "user_id": self.user_id,
            "request_time": self.request_time,
            "client_info": self.client_info,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisRequest':
        """
        Create an AnalysisRequest from dictionary.
        
        Args:
            data: Analysis request data dictionary
            
        Returns:
            AnalysisRequest object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)


class UserActivity(BaseModel):
    """
    User activity model for tracking actions.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    activity_type: str  # view, analysis, recommendation
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "activity_type": "analysis",
                "timestamp": "2023-07-15T10:30:00.000Z",
                "metadata": {
                    "face_shape": "OVAL",
                    "client_info": {
                        "device_type": "mobile",
                        "browser_name": "Chrome"
                    }
                }
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert UserActivity object to dictionary.
        
        Returns:
            dict: User activity data as dictionary
        """
        return {
            "_id": self.id,
            "user_id": self.user_id,
            "activity_type": self.activity_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserActivity':
        """
        Create a UserActivity from dictionary.
        
        Args:
            data: User activity data dictionary
            
        Returns:
            UserActivity object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)


class RecommendationAnalytics(BaseModel):
    """
    Recommendation analytics model for insights.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    face_shape: str
    recommendations_count: int
    recommended_product_ids: List[int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "face_shape": "OVAL",
                "recommendations_count": 5,
                "recommended_product_ids": [123, 456, 789, 321, 654],
                "timestamp": "2023-07-15T10:30:00.000Z"
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RecommendationAnalytics object to dictionary.
        
        Returns:
            dict: Recommendation analytics data as dictionary
        """
        return {
            "_id": self.id,
            "user_id": self.user_id,
            "face_shape": self.face_shape,
            "recommendations_count": self.recommendations_count,
            "recommended_product_ids": self.recommended_product_ids,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendationAnalytics':
        """
        Create a RecommendationAnalytics from dictionary.
        
        Args:
            data: Recommendation analytics data dictionary
            
        Returns:
            RecommendationAnalytics object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)


class AggregateStatistics(BaseModel):
    """
    Aggregate statistics model for dashboard.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    period: str  # daily, weekly, monthly
    date: str
    total_analyses: int = 0
    total_recommendations: int = 0
    face_shape_distribution: Dict[str, int] = {}
    device_distribution: Dict[str, int] = {}
    browser_distribution: Dict[str, int] = {}
    frame_type_distribution: Dict[str, int] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "period": "daily",
                "date": "2023-07-15",
                "total_analyses": 250,
                "total_recommendations": 200,
                "face_shape_distribution": {
                    "OVAL": 100,
                    "ROUND": 75,
                    "SQUARE": 50,
                    "HEART": 25
                },
                "device_distribution": {
                    "mobile": 150,
                    "desktop": 100
                },
                "browser_distribution": {
                    "Chrome": 120,
                    "Safari": 80,
                    "Firefox": 50
                },
                "frame_type_distribution": {
                    "مستطیلی": 80,
                    "گرد": 60,
                    "مربعی": 40,
                    "گربه‌ای": 20
                },
                "created_at": "2023-07-16T00:00:00.000Z"
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AggregateStatistics object to dictionary.
        
        Returns:
            dict: Aggregate statistics data as dictionary
        """
        return {
            "_id": self.id,
            "period": self.period,
            "date": self.date,
            "total_analyses": self.total_analyses,
            "total_recommendations": self.total_recommendations,
            "face_shape_distribution": self.face_shape_distribution,
            "device_distribution": self.device_distribution,
            "browser_distribution": self.browser_distribution,
            "frame_type_distribution": self.frame_type_distribution,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateStatistics':
        """
        Create an AggregateStatistics from dictionary.
        
        Args:
            data: Aggregate statistics data dictionary
            
        Returns:
            AggregateStatistics object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)