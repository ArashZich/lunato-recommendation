from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from database.models.face_analysis import FaceCoordinates, FaceDetection, FaceAnalysis


class FaceRepository:
    """
    Repository for face detection and analysis operations.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize with database connection.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.detections_collection = db.face_detections
        self.analyses_collection = db.face_analyses
    
    # Face Detection methods
    
    async def create_detection(self, detection: FaceDetection) -> str:
        """
        Create a new face detection record.
        
        Args:
            detection: FaceDetection object
            
        Returns:
            str: Created detection ID
        """
        detection_dict = detection.to_dict()
        result = await self.detections_collection.insert_one(detection_dict)
        return str(result.inserted_id)
    
    async def find_detection_by_id(self, detection_id: str) -> Optional[FaceDetection]:
        """
        Find a face detection by ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            FaceDetection object or None if not found
        """
        detection_dict = await self.detections_collection.find_one({"_id": detection_id})
        
        if detection_dict:
            return FaceDetection.from_dict(detection_dict)
        
        return None
    
    async def update_detection(self, detection: FaceDetection) -> bool:
        """
        Update a face detection record.
        
        Args:
            detection: FaceDetection object
            
        Returns:
            bool: True if update successful
        """
        detection_dict = detection.to_dict()
        detection_id = detection_dict.pop("_id")
        
        result = await self.detections_collection.update_one(
            {"_id": detection_id},
            {"$set": detection_dict}
        )
        
        return result.modified_count > 0
    
    async def update_detection_status(self, detection_id: str, status: str, face_shape: Optional[str] = None, error_message: Optional[str] = None) -> bool:
        """
        Update detection status.
        
        Args:
            detection_id: Detection ID
            status: New status (detected, analyzed, error)
            face_shape: Face shape (if analyzed)
            error_message: Error message (if error)
            
        Returns:
            bool: True if update successful
        """
        update_data = {"status": status}
        
        if face_shape:
            update_data["face_shape"] = face_shape
        
        if error_message:
            update_data["error_message"] = error_message
        
        result = await self.detections_collection.update_one(
            {"_id": detection_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def list_detections_by_user(self, user_id: str, limit: int = 10) -> List[FaceDetection]:
        """
        List face detections for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of detections to return
            
        Returns:
            list: List of FaceDetection objects
        """
        detections = []
        cursor = self.detections_collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        async for detection_dict in cursor:
            detections.append(FaceDetection.from_dict(detection_dict))
        
        return detections
    
    # Face Analysis methods
    
    async def create_analysis(self, analysis: FaceAnalysis) -> str:
        """
        Create a new face analysis record.
        
        Args:
            analysis: FaceAnalysis object
            
        Returns:
            str: Created analysis ID
        """
        analysis_dict = analysis.to_dict()
        result = await self.analyses_collection.insert_one(analysis_dict)
        return str(result.inserted_id)
    
    async def find_analysis_by_id(self, analysis_id: str) -> Optional[FaceAnalysis]:
        """
        Find a face analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            FaceAnalysis object or None if not found
        """
        analysis_dict = await self.analyses_collection.find_one({"_id": analysis_id})
        
        if analysis_dict:
            return FaceAnalysis.from_dict(analysis_dict)
        
        return None
    
    async def find_analysis_by_detection_id(self, detection_id: str) -> Optional[FaceAnalysis]:
        """
        Find a face analysis by detection ID.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            FaceAnalysis object or None if not found
        """
        analysis_dict = await self.analyses_collection.find_one({"detection_id": detection_id})
        
        if analysis_dict:
            return FaceAnalysis.from_dict(analysis_dict)
        
        return None
    
    async def update_analysis(self, analysis: FaceAnalysis) -> bool:
        """
        Update a face analysis record.
        
        Args:
            analysis: FaceAnalysis object
            
        Returns:
            bool: True if update successful
        """
        analysis_dict = analysis.to_dict()
        analysis_id = analysis_dict.pop("_id")
        
        result = await self.analyses_collection.update_one(
            {"_id": analysis_id},
            {"$set": analysis_dict}
        )
        
        return result.modified_count > 0
    
    async def list_analyses_by_user(self, user_id: str, limit: int = 10) -> List[FaceAnalysis]:
        """
        List face analyses for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of analyses to return
            
        Returns:
            list: List of FaceAnalysis objects
        """
        analyses = []
        cursor = self.analyses_collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        async for analysis_dict in cursor:
            analyses.append(FaceAnalysis.from_dict(analysis_dict))
        
        return analyses
    
    async def get_latest_analysis_by_user(self, user_id: str) -> Optional[FaceAnalysis]:
        """
        Get the latest face analysis for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            FaceAnalysis object or None if not found
        """
        analysis_dict = await self.analyses_collection.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        if analysis_dict:
            return FaceAnalysis.from_dict(analysis_dict)
        
        return None
    
    async def count_analyses_by_face_shape(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, int]:
        """
        Count analyses by face shape.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            dict: Face shape counts
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
            {"$group": {"_id": "$face_shape", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        counts = {}
        async for result in self.analyses_collection.aggregate(pipeline):
            counts[result["_id"]] = result["count"]
        
        return counts