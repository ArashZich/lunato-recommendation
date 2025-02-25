import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


async def generate_analytics(
    db: AsyncIOMotorDatabase,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    group_by: str = "face_shape"
) -> Dict[str, Any]:
    """
    Generate analytics data.
    
    Args:
        db: MongoDB database
        start_date: Start date for analytics
        end_date: End date for analytics
        group_by: Field to group results by
        
    Returns:
        dict: Analytics data
    """
    try:
        logger.info("Generating analytics")
        
        # Set default dates if not provided
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # Query face analyses
        face_analyses = []
        async for doc in db.face_analyses.find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }):
            face_analyses.append(doc)
            
        # Query recommendations
        recommendations = []
        async for doc in db.recommendations.find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }):
            recommendations.append(doc)
            
        # Calculate total analyses
        total_analyses = len(face_analyses)
        
        if total_analyses == 0:
            return {
                "success": True,
                "message": "No data available for the specified period",
                "total_analyses": 0,
                "face_shapes": [],
                "recommended_frames": [],
                "devices": [],
                "browsers": []
            }
            
        # Calculate face shape distribution
        face_shape_counts = {}
        for analysis in face_analyses:
            face_shape = analysis.get("face_shape")
            if face_shape:
                face_shape_counts[face_shape] = face_shape_counts.get(face_shape, 0) + 1
                
        face_shapes = [
            {
                "face_shape": shape,
                "count": count,
                "percentage": round((count / total_analyses) * 100, 1)
            }
            for shape, count in face_shape_counts.items()
        ]
        
        # Calculate recommended frame types distribution
        frame_type_counts = {}
        for rec in recommendations:
            for frame_type in rec.get("recommended_frame_types", []):
                frame_type_counts[frame_type] = frame_type_counts.get(frame_type, 0) + 1
                
        total_recommendations = sum(frame_type_counts.values())
        
        recommended_frames = [
            {
                "frame_type": frame_type,
                "count": count,
                "percentage": round((count / total_recommendations) * 100, 1) if total_recommendations > 0 else 0
            }
            for frame_type, count in frame_type_counts.items()
        ]
        
        # Calculate device distribution
        device_counts = {}
        for analysis in face_analyses:
            metadata = analysis.get("metadata", {})
            client_info = metadata.get("client_info", {})
            device_type = client_info.get("device_type")
            
            if device_type:
                device_counts[device_type] = device_counts.get(device_type, 0) + 1
                
        devices = [
            {
                "device_type": device_type,
                "count": count,
                "percentage": round((count / total_analyses) * 100, 1)
            }
            for device_type, count in device_counts.items()
        ]
        
        # Calculate browser distribution
        browser_counts = {}
        for analysis in face_analyses:
            metadata = analysis.get("metadata", {})
            client_info = metadata.get("client_info", {})
            browser_name = client_info.get("browser_name")
            
            if browser_name:
                browser_counts[browser_name] = browser_counts.get(browser_name, 0) + 1
                
        browsers = [
            {
                "browser_name": browser,
                "count": count,
                "percentage": round((count / total_analyses) * 100, 1)
            }
            for browser, count in browser_counts.items()
        ]
        
        # Generate time series data
        time_series = await generate_time_series(db, start_date, end_date)
        
        # Generate face shape insights
        face_shape_insights = await generate_face_shape_insights(db, start_date, end_date)
        
        # Generate conversion analytics
        conversion_analytics = await generate_conversion_analytics(db, start_date, end_date)
        
        logger.info("Analytics generation completed")
        
        return {
            "success": True,
            "message": "Analytics generated successfully",
            "total_analyses": total_analyses,
            "face_shapes": sorted(face_shapes, key=lambda x: x["count"], reverse=True),
            "recommended_frames": sorted(recommended_frames, key=lambda x: x["count"], reverse=True),
            "devices": sorted(devices, key=lambda x: x["count"], reverse=True),
            "browsers": sorted(browsers, key=lambda x: x["count"], reverse=True),
            "time_series": time_series,
            "face_shape_insights": face_shape_insights,
            "conversion": conversion_analytics
        }
            
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        return {
            "success": False,
            "message": f"Error generating analytics: {str(e)}"
        }


async def generate_time_series(
    db: AsyncIOMotorDatabase,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Generate time series analytics data.
    
    Args:
        db: MongoDB database
        start_date: Start date
        end_date: End date
        
    Returns:
        dict: Time series data
    """
    try:
        # Aggregate analyses by date
        pipeline = [
            {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        analyses_by_date = []
        async for doc in db.face_analyses.aggregate(pipeline):
            analyses_by_date.append({
                "date": doc["_id"],
                "count": doc["count"]
            })
        
        # Aggregate face shapes by date
        face_shapes_by_date = {}
        
        pipeline = [
            {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "face_shape": "$face_shape"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        
        async for doc in db.face_analyses.aggregate(pipeline):
            date = doc["_id"]["date"]
            face_shape = doc["_id"]["face_shape"]
            count = doc["count"]
            
            if face_shape:
                if face_shape not in face_shapes_by_date:
                    face_shapes_by_date[face_shape] = []
                
                face_shapes_by_date[face_shape].append({
                    "date": date,
                    "count": count
                })
        
        return {
            "analyses_by_date": analyses_by_date,
            "face_shapes_by_date": face_shapes_by_date
        }
        
    except Exception as e:
        logger.error(f"Error generating time series analytics: {str(e)}")
        return {
            "analyses_by_date": [],
            "face_shapes_by_date": {}
        }


async def generate_face_shape_insights(
    db: AsyncIOMotorDatabase,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """
    Generate insights for each face shape.
    
    Args:
        db: MongoDB database
        start_date: Start date
        end_date: End date
        
    Returns:
        list: Face shape insights
    """
    try:
        # Query face analyses
        face_analyses = []
        async for doc in db.face_analyses.find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }):
            face_analyses.append(doc)
            
        # Calculate total analyses
        total_analyses = len(face_analyses)
        
        if total_analyses == 0:
            return []
            
        # Group analyses by face shape
        face_shape_groups = {}
        for analysis in face_analyses:
            face_shape = analysis.get("face_shape")
            if face_shape:
                if face_shape not in face_shape_groups:
                    face_shape_groups[face_shape] = []
                face_shape_groups[face_shape].append(analysis)
        
        # Get recommendations to calculate most common frames
        face_shape_to_frames = {}
        async for rec in db.recommendations.find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }):
            face_shape = rec.get("face_shape")
            if face_shape:
                if face_shape not in face_shape_to_frames:
                    face_shape_to_frames[face_shape] = {}
                    
                for frame_type in rec.get("recommended_frame_types", []):
                    face_shape_to_frames[face_shape][frame_type] = face_shape_to_frames[face_shape].get(frame_type, 0) + 1
        
        # Calculate insights for each face shape
        insights = []
        for face_shape, analyses in face_shape_groups.items():
            count = len(analyses)
            percentage = round((count / total_analyses) * 100, 1)
            
            # Calculate average confidence
            confidences = [a.get("confidence", 0) for a in analyses if "confidence" in a]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Get most recommended frames
            frame_counts = face_shape_to_frames.get(face_shape, {})
            most_recommended = [
                frame for frame, _ in sorted(
                    frame_counts.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )[:3]
            ]
            
            insights.append({
                "face_shape": face_shape,
                "count": count,
                "percentage": percentage,
                "most_recommended_frames": most_recommended,
                "average_confidence": round(avg_confidence, 1)
            })
        
        return sorted(insights, key=lambda x: x["count"], reverse=True)
            
    except Exception as e:
        logger.error(f"Error generating face shape insights: {str(e)}")
        return []


async def generate_conversion_analytics(
    db: AsyncIOMotorDatabase,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Generate conversion analytics.
    
    Args:
        db: MongoDB database
        start_date: Start date
        end_date: End date
        
    Returns:
        dict: Conversion analytics
    """
    try:
        # Count requests
        total_requests = await db.analysis_requests.count_documents({
            "request_time": {"$gte": start_date, "$lte": end_date}
        })
        
        # Count completed analyses
        completed_analyses = await db.face_analyses.count_documents({
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        # Count recommendations
        recommendations = await db.recommendations.count_documents({
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        # Calculate conversion rate
        conversion_rate = (completed_analyses / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate average recommendations per analysis
        avg_recommendations = recommendations / completed_analyses if completed_analyses > 0 else 0
        
        return {
            "total_requests": total_requests,
            "completed_analyses": completed_analyses,
            "viewed_recommendations": recommendations,
            "conversion_rate": round(conversion_rate, 1),
            "average_recommendations": round(avg_recommendations, 1)
        }
            
    except Exception as e:
        logger.error(f"Error generating conversion analytics: {str(e)}")
        return {
            "total_requests": 0,
            "completed_analyses": 0,
            "viewed_recommendations": 0,
            "conversion_rate": 0,
            "average_recommendations": 0
        }


async def track_user_activity(
    db: AsyncIOMotorDatabase, 
    user_id: str, 
    activity_type: str, 
    metadata: Dict[str, Any]
) -> bool:
    """
    Track user activity for analytics.
    
    Args:
        db: MongoDB database
        user_id: User ID
        activity_type: Type of activity (view, analysis, recommendation)
        metadata: Additional metadata
        
    Returns:
        bool: Success status
    """
    try:
        # Create activity record
        activity = {
            "user_id": user_id,
            "activity_type": activity_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        
        # Insert into analytics collection
        await db.user_activities.insert_one(activity)
        
        return True
        
    except Exception as e:
        logger.error(f"Error tracking user activity: {str(e)}")
        return False