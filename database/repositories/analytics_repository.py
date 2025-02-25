from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

from database.models.analytics import AnalysisRequest, UserActivity, AggregateStatistics


class AnalyticsRepository:
    """
    Repository for analytics operations.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize with database connection.
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.requests_collection = db.analysis_requests
        self.user_activities_collection = db.user_activities
        self.stats_collection = db.aggregate_statistics
    
    # Request tracking methods
    
    async def record_request(self, user_id: str, client_info: Dict[str, Any]) -> str:
        """
        Record an analysis request.
        
        Args:
            user_id: User ID
            client_info: Client device information
            
        Returns:
            str: Created request ID
        """
        request = {
            "user_id": user_id,
            "request_time": datetime.utcnow(),
            "client_info": client_info,
            "status": "processing"
        }
        
        result = await self.requests_collection.insert_one(request)
        return str(result.inserted_id)
    
    async def update_request_status(self, request_id: str, status: str) -> bool:
        """
        Update a request status.
        
        Args:
            request_id: Request ID
            status: New status (processing, completed, error)
            
        Returns:
            bool: True if update successful
        """
        result = await self.requests_collection.update_one(
            {"_id": request_id},
            {"$set": {"status": status}}
        )
        
        return result.modified_count > 0
    
    async def count_requests(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
        """
        Count analysis requests.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            int: Request count
        """
        query = {}
        
        if start_date or end_date:
            query["request_time"] = {}
            
            if start_date:
                query["request_time"]["$gte"] = start_date
                
            if end_date:
                query["request_time"]["$lte"] = end_date
        
        return await self.requests_collection.count_documents(query)
    
    async def count_completed_requests(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
        """
        Count completed analysis requests.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            int: Completed request count
        """
        query = {"status": "completed"}
        
        if start_date or end_date:
            query["request_time"] = {}
            
            if start_date:
                query["request_time"]["$gte"] = start_date
                
            if end_date:
                query["request_time"]["$lte"] = end_date
        
        return await self.requests_collection.count_documents(query)
    
    # User activity methods
    
    async def track_activity(self, user_id: str, activity_type: str, metadata: Dict[str, Any] = {}) -> str:
        """
        Track a user activity.
        
        Args:
            user_id: User ID
            activity_type: Type of activity (view, analysis, recommendation)
            metadata: Additional metadata
            
        Returns:
            str: Created activity ID
        """
        activity = {
            "user_id": user_id,
            "activity_type": activity_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        
        result = await self.user_activities_collection.insert_one(activity)
        return str(result.inserted_id)
    
    async def get_user_activities(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get activities for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of activities to return
            
        Returns:
            list: List of activities
        """
        activities = []
        cursor = self.user_activities_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        
        async for activity in cursor:
            # Convert ObjectId to string
            activity["_id"] = str(activity["_id"])
            activities.append(activity)
        
        return activities
    
    async def count_activities_by_type(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, int]:
        """
        Count activities by type.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            dict: Activity type counts
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
            {"$group": {"_id": "$activity_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        counts = {}
        async for result in self.user_activities_collection.aggregate(pipeline):
            counts[result["_id"]] = result["count"]
        
        return counts
    
    # Aggregate statistics methods
    
    async def generate_daily_statistics(self, date: datetime) -> str:
        """
        Generate and store daily aggregate statistics.
        
        Args:
            date: Date to generate statistics for
            
        Returns:
            str: Created statistics ID
        """
        # Set date range for the day
        start_date = datetime(date.year, date.month, date.day)
        end_date = start_date + timedelta(days=1)
        
        # Get face shape distribution
        face_shape_distribution = await self._get_face_shape_distribution(start_date, end_date)
        
        # Get device distribution
        device_distribution = await self._get_device_distribution(start_date, end_date)
        
        # Get browser distribution
        browser_distribution = await self._get_browser_distribution(start_date, end_date)
        
        # Get frame type distribution
        frame_type_distribution = await self._get_frame_type_distribution(start_date, end_date)
        
        # Count analyses and recommendations
        total_analyses = await self.db.face_analyses.count_documents({
            "created_at": {"$gte": start_date, "$lt": end_date}
        })
        
        total_recommendations = await self.db.recommendations.count_documents({
            "created_at": {"$gte": start_date, "$lt": end_date}
        })
        
        # Create statistics document
        stats = {
            "period": "daily",
            "date": start_date.strftime("%Y-%m-%d"),
            "total_analyses": total_analyses,
            "total_recommendations": total_recommendations,
            "face_shape_distribution": face_shape_distribution,
            "device_distribution": device_distribution,
            "browser_distribution": browser_distribution,
            "frame_type_distribution": frame_type_distribution,
            "created_at": datetime.utcnow()
        }
        
        # Check if stats already exist for this day
        existing = await self.stats_collection.find_one({
            "period": "daily",
            "date": start_date.strftime("%Y-%m-%d")
        })
        
        if existing:
            # Update existing stats
            await self.stats_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": stats}
            )
            return str(existing["_id"])
        else:
            # Insert new stats
            result = await self.stats_collection.insert_one(stats)
            return str(result.inserted_id)
    
    async def get_statistics(self, period: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get aggregate statistics for a period.
        
        Args:
            period: Period type (daily, weekly, monthly)
            start_date: Start date
            end_date: End date
            
        Returns:
            list: List of statistics
        """
        stats = []
        cursor = self.stats_collection.find({
            "period": period,
            "date": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }).sort("date", 1)
        
        async for stat in cursor:
            # Convert ObjectId to string
            stat["_id"] = str(stat["_id"])
            stats.append(stat)
        
        return stats
    
    async def get_latest_statistics(self, period: str = "daily") -> Optional[Dict[str, Any]]:
        """
        Get the latest statistics for a period.
        
        Args:
            period: Period type (daily, weekly, monthly)
            
        Returns:
            dict: Statistics or None if not found
        """
        stats = await self.stats_collection.find_one(
            {"period": period},
            sort=[("date", -1)]
        )
        
        if stats:
            # Convert ObjectId to string
            stats["_id"] = str(stats["_id"])
            return stats
        
        return None
    
    # Helper methods for statistics generation
    
    async def _get_face_shape_distribution(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """
        Get face shape distribution for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Face shape counts
        """
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lt": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$face_shape",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        result = {}
        async for doc in self.db.face_analyses.aggregate(pipeline):
            if doc["_id"]:  # Skip null values
                result[doc["_id"]] = doc["count"]
        
        return result
    
    async def _get_device_distribution(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """
        Get device distribution for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Device type counts
        """
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lt": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$metadata.client_info.device_type",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        result = {}
        async for doc in self.db.face_analyses.aggregate(pipeline):
            if doc["_id"]:  # Skip null values
                result[doc["_id"]] = doc["count"]
        
        return result
    
    async def _get_browser_distribution(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """
        Get browser distribution for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Browser counts
        """
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lt": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$metadata.client_info.browser_name",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        result = {}
        async for doc in self.db.face_analyses.aggregate(pipeline):
            if doc["_id"]:  # Skip null values
                result[doc["_id"]] = doc["count"]
        
        return result
    
    async def _get_frame_type_distribution(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """
        Get frame type distribution for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Frame type counts
        """
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lt": end_date}
                }
            },
            {
                "$unwind": "$recommended_frame_types"
            },
            {
                "$group": {
                    "_id": "$recommended_frame_types",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        result = {}
        async for doc in self.db.recommendations.aggregate(pipeline):
            if doc["_id"]:  # Skip null values
                result[doc["_id"]] = doc["count"]
        
        return result
    
    # Time series analytics methods
    
    async def get_time_series_data(self, interval: str, field: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get time series data for a field.
        
        Args:
            interval: Time interval (day, week, month)
            field: Field to aggregate (analyses, recommendations)
            start_date: Start date
            end_date: End date
            
        Returns:
            list: List of time series data points
        """
        # Determine the collection based on the field
        if field == "analyses":
            collection = self.db.face_analyses
        elif field == "recommendations":
            collection = self.db.recommendations
        else:
            return []
        
        # Determine the date format based on the interval
        if interval == "day":
            date_format = "%Y-%m-%d"
        elif interval == "week":
            date_format = "%Y-%U"  # Year and week number
        elif interval == "month":
            date_format = "%Y-%m"
        else:
            return []
        
        # Aggregate data by interval
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": date_format, "date": "$created_at"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        result = []
        async for doc in collection.aggregate(pipeline):
            result.append({
                "date": doc["_id"],
                "count": doc["count"]
            })
        
        return result
    
    async def get_face_shape_time_series(self, interval: str, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get time series data for face shapes.
        
        Args:
            interval: Time interval (day, week, month)
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Dictionary of face shapes with time series data
        """
        # Determine the date format based on the interval
        if interval == "day":
            date_format = "%Y-%m-%d"
        elif interval == "week":
            date_format = "%Y-%U"  # Year and week number
        elif interval == "month":
            date_format = "%Y-%m"
        else:
            return {}
        
        # Aggregate data by interval and face shape
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "date": {"$dateToString": {"format": date_format, "date": "$created_at"}},
                        "face_shape": "$face_shape"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id.date": 1}
            }
        ]
        
        # Group results by face shape
        result = {}
        async for doc in self.db.face_analyses.aggregate(pipeline):
            date = doc["_id"]["date"]
            face_shape = doc["_id"]["face_shape"]
            count = doc["count"]
            
            if face_shape:  # Skip null values
                if face_shape not in result:
                    result[face_shape] = []
                
                result[face_shape].append({
                    "date": date,
                    "count": count
                })
        
        return result
    
    # Conversion analytics methods
    
    async def get_conversion_analytics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get conversion analytics.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            dict: Conversion analytics data
        """
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Count total analysis requests
        total_requests = await self.count_requests(start_date, end_date)
        
        # Count completed analyses
        completed_analyses = await self.db.face_analyses.count_documents({
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        # Count recommendations
        viewed_recommendations = await self.db.recommendations.count_documents({
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        # Calculate conversion rates
        analysis_rate = (completed_analyses / total_requests * 100) if total_requests > 0 else 0
        recommendation_rate = (viewed_recommendations / completed_analyses * 100) if completed_analyses > 0 else 0
        
        # Calculate average recommendations per analysis
        avg_recommendations = viewed_recommendations / completed_analyses if completed_analyses > 0 else 0
        
        return {
            "total_requests": total_requests,
            "completed_analyses": completed_analyses,
            "viewed_recommendations": viewed_recommendations,
            "analysis_rate": round(analysis_rate, 1),
            "recommendation_rate": round(recommendation_rate, 1),
            "average_recommendations": round(avg_recommendations, 1)
        }