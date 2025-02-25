from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from database.models.user import User


class UserRepository:
    """
    Repository for User model operations.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize with database connection.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.users
    
    async def create(self, user: User) -> str:
        """
        Create a new user.
        
        Args:
            user: User object
            
        Returns:
            str: Created user ID
        """
        user_dict = user.to_dict()
        result = await self.collection.insert_one(user_dict)
        return str(result.inserted_id)
    
    async def find_by_id(self, user_id: str) -> Optional[User]:
        """
        Find a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        user_dict = await self.collection.find_one({"_id": user_id})
        
        if user_dict:
            return User.from_dict(user_dict)
        
        return None
    
    async def find_by_device_id(self, device_id: str) -> Optional[User]:
        """
        Find a user by device ID.
        
        Args:
            device_id: Device ID
            
        Returns:
            User object or None if not found
        """
        user_dict = await self.collection.find_one({"device_id": device_id})
        
        if user_dict:
            return User.from_dict(user_dict)
        
        return None
    
    async def update(self, user: User) -> bool:
        """
        Update a user.
        
        Args:
            user: User object
            
        Returns:
            bool: True if update successful
        """
        user_dict = user.to_dict()
        user_id = user_dict.pop("_id")
        
        result = await self.collection.update_one(
            {"_id": user_id},
            {"$set": user_dict}
        )
        
        return result.modified_count > 0
    
    async def update_last_active(self, user_id: str) -> bool:
        """
        Update user's last active timestamp.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if update successful
        """
        result = await self.collection.update_one(
            {"_id": user_id},
            {"$set": {"last_active": datetime.utcnow()}}
        )
        
        return result.modified_count > 0
    
    async def delete(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if delete successful
        """
        result = await self.collection.delete_one({"_id": user_id})
        return result.deleted_count > 0
    
    async def list(self, skip: int = 0, limit: int = 100) -> List[User]:
        """
        List users with pagination.
        
        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            list: List of User objects
        """
        users = []
        cursor = self.collection.find().skip(skip).limit(limit)
        
        async for user_dict in cursor:
            users.append(User.from_dict(user_dict))
        
        return users
    
    async def count(self) -> int:
        """
        Count users.
        
        Returns:
            int: User count
        """
        return await self.collection.count_documents({})
    
    async def find_or_create_by_device_id(self, device_id: str, data: Dict[str, Any]) -> User:
        """
        Find a user by device ID or create if not found.
        
        Args:
            device_id: Device ID
            data: User data
            
        Returns:
            User object
        """
        user = await self.find_by_device_id(device_id)
        
        if not user:
            # Create a new user
            user = User(device_id=device_id, **data)
            await self.create(user)
        
        return user