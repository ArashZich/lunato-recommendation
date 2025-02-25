from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class User(BaseModel):
    """
    User model for database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    browser_name: Optional[str] = None
    browser_version: Optional[str] = None
    language: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "device_id": "b642b4217b34b1e8d3bd915fc65c4452",
                "device_type": "mobile",
                "os_name": "iOS",
                "os_version": "15.4",
                "browser_name": "Safari",
                "browser_version": "15.4",
                "language": "fa-IR",
                "ip_address": "192.168.1.1",
                "created_at": "2023-07-15T10:30:00.000Z",
                "last_active": "2023-07-15T10:30:00.000Z"
            }
        }
    
    @classmethod
    def from_client_info(cls, client_info: Dict[str, Any]) -> 'User':
        """
        Create a User from client info.
        
        Args:
            client_info: Client info dictionary
            
        Returns:
            User object
        """
        return cls(
            device_id=client_info.get("device_id"),
            device_type=client_info.get("device_type"),
            os_name=client_info.get("os_name"),
            os_version=client_info.get("os_version"),
            browser_name=client_info.get("browser_name"),
            browser_version=client_info.get("browser_version"),
            language=client_info.get("language"),
            ip_address=client_info.get("ip_address")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert User object to dictionary.
        
        Returns:
            dict: User data as dictionary
        """
        return {
            "_id": self.id,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "browser_name": self.browser_name,
            "browser_version": self.browser_version,
            "language": self.language,
            "ip_address": self.ip_address,
            "created_at": self.created_at,
            "last_active": self.last_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Create a User from dictionary.
        
        Args:
            data: User data dictionary
            
        Returns:
            User object
        """
        # Convert _id to id if present
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
            
        return cls(**data)