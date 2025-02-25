from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from database.models.recommendation import Recommendation, RecommendedFrame


class RecommendationRepository:
    """
    Repository for recommendation operations.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize with database connection.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.recommendations
        self.analytics_collection = db.recommendation_analytics
    
    async def create(self, recommendation: Recommendation) -> str:
        """
        Create a new recommendation record.
        
        Args:
            recommendation: Recommendation object
            
        Returns:
            str: Created recommendation ID
        """
        recommendation_dict = recommendation.to_dict()
        result = await self.collection.insert_one(recommendation_dict)
        return str(result.inserted_id)
    
    async def find_by_id(self, recommendation_id: str) -> Optional[Recommendation]:
        """
        Find a recommendation by ID.
        
        Args:
            recommendation_id: Recommendation ID
            
        Returns:
            Recommendation object or None if not found
        """
        recommendation_dict = await self.collection.find_one({"_id": recommendation_id})
        
        if recommendation_dict:
            return Recommendation.from_dict(recommendation_dict)
        
        return None
    
    async def find_by_detection_id(self, detection_id: str) -> Optional[Recommendation]:
        """
        Find a recommendation by detection ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            Recommendation object or None if not found
        """
        recommendation_dict = await self.collection.find_one({"detection_id": detection_id})
        
        if recommendation_dict:
            return Recommendation.from_dict(recommendation_dict)
        
        return None
    
    async def find_latest_by_user_and_face_shape(self, user_id: str, face_shape: str) -> Optional[Recommendation]:
        """
        Find the latest recommendation for a user and face shape.
        
        Args:
            user_id: User ID
            face_shape: Face shape
            
        Returns:
            Recommendation object or None if not found
        """
        recommendation_dict = await self.collection.find_one(
            {"user_id": user_id, "face_shape": face_shape},
            sort=[("created_at", -1)]
        )
        
        if recommendation_dict:
            return Recommendation.from_dict(recommendation_dict)
        
        return None
    
    async def update(self, recommendation: Recommendation) -> bool:
        """
        Update a recommendation record.
        
        Args:
            recommendation: Recommendation object
            
        Returns:
            bool: True if update successful
        """
        recommendation_dict = recommendation.to_dict()
        recommendation_id = recommendation_dict.pop("_id")
        
        result = await self.collection.update_one(
            {"_id": recommendation_id},
            {"$set": recommendation_dict}
        )
        
        return result.modified_count > 0
    
    async def list_by_user(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """
        List recommendations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of recommendations to return
            
        Returns:
            list: List of Recommendation objects
        """
        recommendations = []
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        async for recommendation_dict in cursor:
            recommendations.append(Recommendation.from_dict(recommendation_dict))
        
        return recommendations
    
    async def get_latest_by_user(self, user_id: str) -> Optional[Recommendation]:
        """
        Get the latest recommendation for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Recommendation object or None if not found
        """
        recommendation_dict = await self.collection.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        if recommendation_dict:
            return Recommendation.from_dict(recommendation_dict)
        
        return None
    
    async def count_by_frame_type(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, int]:
        """
        Count recommendations by frame type.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            dict: Frame type counts
        """
        query = {}
        
        if start_date or end_date:
            query["created_at"] = {}
            
            if start_date:
                query["created_at"]["$gte"] = start_date
                
            if end_date:
                query["created_at"]["$lte"] = end_date
        
        pipeline = [
            {"$match": query},
            {"$unwind": "$recommended_frame_types"},
            {"$group": {"_id": "$recommended_frame_types", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        counts = {}
        async for result in self.collection.aggregate(pipeline):
            counts[result["_id"]] = result["count"]
        
        return counts
    
    async def record_analytics(self, user_id: str, face_shape: str, product_ids: List[int]) -> str:
        """
        Record recommendation analytics.
        
        Args:
            user_id: User ID
            face_shape: Face shape
            product_ids: Recommended product IDs
            
        Returns:
            str: Created analytics record ID
        """
        analytics = {
            "user_id": user_id,
            "face_shape": face_shape,
            "recommendations_count": len(product_ids),
            "recommended_product_ids": product_ids,
            "timestamp": datetime.utcnow()
        }
        
        result = await self.analytics_collection.insert_one(analytics)
        return str(result.inserted_id)
    
    async def get_popular_frames(self, limit: int = 10, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get the most popular recommended frames.
        
        Args:
            limit: Maximum number of frames to return
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            list: List of popular frames with counts
        """
        query = {}
        
        if start_date or end_date:
            query["timestamp"] = {}
            
            if start_date:
                query["timestamp"]["$gte"] = start_date
                
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        pipeline = [
            {"$match": query},
            {"$unwind": "$recommended_product_ids"},
            {"$group": {"_id": "$recommended_product_ids", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        
        popular_frames = []
        async for result in self.analytics_collection.aggregate(pipeline):
            popular_frames.append({
                "product_id": result["_id"],
                "count": result["count"]
            })
        
        return popular_frames
    
    async def get_frame_popularity_by_face_shape(self, face_shape: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most popular frames for a specific face shape.
        
        Args:
            face_shape: Face shape
            limit: Maximum number of frames to return
            
        Returns:
            list: List of popular frames with counts
        """
        pipeline = [
            {"$match": {"face_shape": face_shape}},
            {"$unwind": "$recommended_product_ids"},
            {"$group": {"_id": "$recommended_product_ids", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        
        popular_frames = []
        async for result in self.analytics_collection.aggregate(pipeline):
            popular_frames.append({
                "product_id": result["_id"],
                "count": result["count"]
            })
        
        return popular_frames