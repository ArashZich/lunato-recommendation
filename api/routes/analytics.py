from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from api.schemas.request import AnalyticsRequest
from api.schemas.response import AnalyticsResponse, FaceShapeCount, FrameTypeCount, DeviceAnalytics, BrowserAnalytics
from api.schemas.analytics import AggregateAnalytics, TimeSeriesAnalytics, ConversionAnalytics, FaceShapeInsights
from database.connection import get_database
from api.services.analytics import generate_analytics

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics(
    request: Optional[AnalyticsRequest] = None,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get analytics data for face shapes and recommendations.
    
    This endpoint returns aggregated statistics about:
    - Face shape distribution
    - Recommended frame types
    - Device and browser usage
    """
    try:
        if not request:
            request = AnalyticsRequest()
            
        # Set default dates if not provided
        end_date = request.end_date or datetime.utcnow()
        start_date = request.start_date or (end_date - timedelta(days=30))
        
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
            return AnalyticsResponse(
                success=True,
                message="No data available for the specified period",
                total_analyses=0,
                face_shapes=[],
                recommended_frames=[],
                devices=[],
                browsers=[]
            )
            
        # Calculate face shape distribution
        face_shape_counts = {}
        for analysis in face_analyses:
            face_shape = analysis.get("face_shape")
            if face_shape:
                face_shape_counts[face_shape] = face_shape_counts.get(face_shape, 0) + 1
                
        face_shapes = [
            FaceShapeCount(
                face_shape=shape,
                count=count,
                percentage=round((count / total_analyses) * 100, 1)
            )
            for shape, count in face_shape_counts.items()
        ]
        
        # Calculate recommended frame types distribution
        frame_type_counts = {}
        for rec in recommendations:
            for frame_type in rec.get("recommended_frame_types", []):
                frame_type_counts[frame_type] = frame_type_counts.get(frame_type, 0) + 1
                
        total_recommendations = sum(frame_type_counts.values())
        
        recommended_frames = [
            FrameTypeCount(
                frame_type=frame_type,
                count=count,
                percentage=round((count / total_recommendations) * 100, 1) if total_recommendations > 0 else 0
            )
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
            DeviceAnalytics(
                device_type=device_type,
                count=count,
                percentage=round((count / total_analyses) * 100, 1)
            )
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
            BrowserAnalytics(
                browser_name=browser,
                count=count,
                percentage=round((count / total_analyses) * 100, 1)
            )
            for browser, count in browser_counts.items()
        ]
        
        return AnalyticsResponse(
            success=True,
            message="Analytics generated successfully",
            total_analyses=total_analyses,
            face_shapes=sorted(face_shapes, key=lambda x: x.count, reverse=True),
            recommended_frames=sorted(recommended_frames, key=lambda x: x.count, reverse=True),
            devices=sorted(devices, key=lambda x: x.count, reverse=True),
            browsers=sorted(browsers, key=lambda x: x.count, reverse=True)
        )
            
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")


@router.get("/analytics/face-shapes", tags=["Analytics"])
async def get_face_shape_insights(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get detailed insights about face shapes.
    
    This endpoint returns detailed stats about each face shape including:
    - Frequency and distribution
    - Most commonly recommended frames
    - Average confidence scores
    """
    try:
        # Set default dates if not provided
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # Query face analyses
        face_analyses = []
        async for doc in db.face_analyses.find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }):
            face_analyses.append(doc)
            
        # Calculate total analyses
        total_analyses = len(face_analyses)
        
        if total_analyses == 0:
            return {
                "success": True,
                "message": "No data available for the specified period",
                "face_shape_insights": []
            }
            
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
        
        return {
            "success": True,
            "message": "Face shape insights generated successfully",
            "face_shape_insights": sorted(insights, key=lambda x: x["count"], reverse=True)
        }
            
    except Exception as e:
        logger.error(f"Error generating face shape insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating face shape insights: {str(e)}")


@router.get("/analytics/time-series", tags=["Analytics"])
async def get_time_series_analytics(
    period: str = Query("day", description="Aggregation period (day, week, month)"),
    days: int = Query(30, description="Number of days to include"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get time series analytics data.
    
    This endpoint returns data showing trends over time:
    - Number of analyses per day/week/month
    - Face shape distribution over time
    """
    try:
        # Set date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Define group by format based on period
        if period == "week":
            date_format = "%Y-%U"  # Year and week number
            date_trunc = {"$dateToString": {"format": "%Y-%U", "date": "$created_at"}}
        elif period == "month":
            date_format = "%Y-%m"  # Year and month
            date_trunc = {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}}
        else:  # default to day
            date_format = "%Y-%m-%d"  # Year, month, day
            date_trunc = {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}
        
        # Aggregate analyses by date
        pipeline = [
            {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
            {"$group": {
                "_id": date_trunc,
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
                    "date": date_trunc,
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
            "success": True,
            "message": "Time series analytics generated successfully",
            "period": period,
            "analyses_by_date": analyses_by_date,
            "face_shapes_by_date": face_shapes_by_date
        }
            
    except Exception as e:
        logger.error(f"Error generating time series analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating time series analytics: {str(e)}")