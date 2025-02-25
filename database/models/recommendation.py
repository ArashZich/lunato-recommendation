from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class RecommendedFrame(BaseModel):
    """
    Recommended eyeglass frame model.
    """
    id: int
    name: str
    permalink: str
    price: str
    regular_price: Optional[str] = None
    frame_type: str
    images: List[str]
    match_score: float


class Recommendation(BaseModel):
    """
    Recommendation model for database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    detection_id: Optional[str] = None
    face_shape: str
    recommended_frame_types: List[str]
    recommendations: List[RecommendedFrame]
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "detection_id": "550e8400-e29b-41d4-a716-446655440002",
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
                ],
                "metadata": {
                    "client_info": {
                        "device_type": "mobile",
                        "browser_name": "Chrome"
                    }
                },
                "created_at": "2023-07-15T10:30:00.000Z"
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Recommendation object to dictionary.
        
        Returns:
            dict: Recommendation data as dictionary
        """
        recommendations_dict = []
        for rec in self.recommendations:
            recommendations_dict.append(rec.dict())
            
        return {
            "_id": self.id,
            "user_id": self.user_id,
            "detection_id": self.detection_id,
            "face_shape": self.face_shape,
            "recommended_frame_types": self.recommended_frame_types,
            "recommendations": recommendations_dict,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """
        Create a Recommendation from dictionary.
        
        Args:
            data: Recommendation data dictionary
            
        Returns:
            Recommendation object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
        
        # Convert recommendations to RecommendedFrame objects
        if "recommendations" in data and isinstance(data["recommendations"], list):
            data["recommendations"] = [RecommendedFrame(**rec) for rec in data["recommendations"]]
            
        return cls(**data)