from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class FaceCoordinates(BaseModel):
    """
    Face coordinates model.
    """
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    aspect_ratio: float


class FaceDetection(BaseModel):
    """
    Face detection model for database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    face_coordinates: FaceCoordinates
    image_width: int
    image_height: int
    status: str = "detected"  # detected, analyzed, error
    face_shape: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "550e8400-e29b-41d4-a716-446655440001",
                "face_coordinates": {
                    "x": 100,
                    "y": 80,
                    "width": 200,
                    "height": 220,
                    "center_x": 200,
                    "center_y": 190,
                    "aspect_ratio": 0.91
                },
                "image_width": 640,
                "image_height": 480,
                "status": "analyzed",
                "face_shape": "OVAL",
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
        Convert FaceDetection object to dictionary.
        
        Returns:
            dict: Face detection data as dictionary
        """
        return {
            "_id": self.id,
            "user_id": self.user_id,
            "face_coordinates": self.face_coordinates.dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status,
            "face_shape": self.face_shape,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceDetection':
        """
        Create a FaceDetection from dictionary.
        
        Args:
            data: Face detection data dictionary
            
        Returns:
            FaceDetection object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
        
        # Convert face_coordinates to FaceCoordinates object
        if "face_coordinates" in data and isinstance(data["face_coordinates"], dict):
            data["face_coordinates"] = FaceCoordinates(**data["face_coordinates"])
            
        return cls(**data)


class FaceAnalysis(BaseModel):
    """
    Face analysis model for database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str
    user_id: str
    face_shape: str
    confidence: float
    shape_metrics: Dict[str, float]
    features: Dict[str, float]
    status: str = "completed"  # completed, error
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "detection_id": "550e8400-e29b-41d4-a716-446655440001",
                "user_id": "550e8400-e29b-41d4-a716-446655440002",
                "face_shape": "OVAL",
                "confidence": 85.5,
                "shape_metrics": {
                    "OVAL": 5,
                    "ROUND": 3,
                    "SQUARE": 1
                },
                "features": {
                    "width_to_length_ratio": 0.75,
                    "cheekbone_to_jaw_ratio": 1.05,
                    "jaw_angle": 130
                },
                "status": "completed",
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
        Convert FaceAnalysis object to dictionary.
        
        Returns:
            dict: Face analysis data as dictionary
        """
        return {
            "_id": self.id,
            "detection_id": self.detection_id,
            "user_id": self.user_id,
            "face_shape": self.face_shape,
            "confidence": self.confidence,
            "shape_metrics": self.shape_metrics,
            "features": self.features,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceAnalysis':
        """
        Create a FaceAnalysis from dictionary.
        
        Args:
            data: Face analysis data dictionary
            
        Returns:
            FaceAnalysis object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)