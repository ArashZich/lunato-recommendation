from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from typing import Optional, List
from datetime import datetime

from api.schemas.request import FrameRecommendationRequest
from api.schemas.response import FrameRecommendationResponse, RecommendedFrame
from api.utils.client_info import extract_client_info
from database.connection import get_database
from celery_app.tasks.frame_matching import match_frames

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommendations", response_model=FrameRecommendationResponse, tags=["Frame Recommendations"])
async def get_recommendations(
    request: FrameRecommendationRequest,
    req: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get frame recommendations based on face shape.
    
    This endpoint can be used in two ways:
    1. Provide a face_shape to get recommendations
    2. Provide a detection_id from a previous face analysis
    """
    try:
        user_id = request.user_id
        
        # Check if we have a valid face shape or detection ID
        if not request.face_shape and not request.detection_id:
            raise HTTPException(
                status_code=400,
                detail="Either face_shape or detection_id must be provided"
            )
            
        face_shape = request.face_shape
        
        # If detection_id is provided, get the face shape from the analysis
        if request.detection_id:
            # Check if the analysis exists
            analysis = await db.face_analyses.find_one({"detection_id": request.detection_id})
            
            if not analysis:
                # Check if the detection exists but analysis is not complete
                detection = await db.face_detections.find_one({"_id": request.detection_id})
                
                if not detection:
                    raise HTTPException(status_code=404, detail="Face analysis not found")
                
                if detection.get("status") != "analyzed":
                    return FrameRecommendationResponse(
                        success=False,
                        message=f"Face analysis not complete. Status: {detection.get('status', 'processing')}"
                    )
                    
                face_shape = detection.get("face_shape")
            else:
                face_shape = analysis.get("face_shape")
                
            # If we still don't have a face shape, return an error
            if not face_shape:
                raise HTTPException(
                    status_code=400,
                    detail="Face shape not available. Analysis may be in progress."
                )
        
        # Get recommendations from the database if they exist
        if user_id:
            existing_recommendations = await db.recommendations.find_one(
                {"user_id": user_id, "face_shape": face_shape},
                sort=[("created_at", -1)]  # Get the most recent
            )
            
            if existing_recommendations:
                # Apply price filters if requested
                recommendations = existing_recommendations.get("recommendations", [])
                
                if request.filter_by_price and (request.min_price is not None or request.max_price is not None):
                    recommendations = [
                        r for r in recommendations 
                        if (request.min_price is None or float(r.get("price", 0)) >= request.min_price) and
                           (request.max_price is None or float(r.get("price", 0)) <= request.max_price)
                    ]
                
                # Limit results if requested
                if request.limit and request.limit > 0:
                    recommendations = recommendations[:request.limit]
                
                return FrameRecommendationResponse(
                    success=True,
                    message="Recommendations retrieved successfully",
                    face_shape=face_shape,
                    recommended_frame_types=existing_recommendations.get("recommended_frame_types", []),
                    recommendations=recommendations
                )
        
        # If no existing recommendations, generate new ones
        # Extract client info for analytics
        client_info = extract_client_info(req)
        
        # Create metadata
        metadata = {
            "client_info": client_info.dict() if client_info else {},
            "timestamp": datetime.utcnow().isoformat(),
            "request_ip": req.client.host,
            "filter_by_price": request.filter_by_price,
            "min_price": request.min_price,
            "max_price": request.max_price
        }
        
        # Generate new recommendations using the frame matching task
        task = match_frames.delay(
            request.detection_id if request.detection_id else None,
            face_shape,
            user_id if user_id else "anonymous",
            metadata
        )
        
        # Wait for the task to complete with a timeout
        result = task.get(timeout=30)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to generate recommendations")
            )
            
        # Apply price filters if requested
        recommendations = result.get("recommendations", [])
        
        if request.filter_by_price and (request.min_price is not None or request.max_price is not None):
            recommendations = [
                r for r in recommendations 
                if (request.min_price is None or float(r.get("price", 0)) >= request.min_price) and
                   (request.max_price is None or float(r.get("price", 0)) <= request.max_price)
            ]
        
        # Limit results if requested
        if request.limit and request.limit > 0:
            recommendations = recommendations[:request.limit]
        
        return FrameRecommendationResponse(
            success=True,
            message="Recommendations generated successfully",
            face_shape=face_shape,
            recommended_frame_types=result.get("recommended_frame_types", []),
            recommendations=recommendations
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.get("/recommendations/face-shape/{face_shape}", response_model=FrameRecommendationResponse, tags=["Frame Recommendations"])
async def get_recommendations_by_face_shape(
    face_shape: str,
    limit: Optional[int] = Query(10, description="Maximum number of recommendations to return"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    req: Request = None,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get frame recommendations for a specific face shape.
    
    This endpoint allows getting recommendations by directly specifying a face shape.
    """
    # Create a FrameRecommendationRequest object
    request = FrameRecommendationRequest(
        face_shape=face_shape.upper(),
        filter_by_price=min_price is not None or max_price is not None,
        min_price=min_price,
        max_price=max_price,
        limit=limit
    )
    
    # Call the main recommendations endpoint
    return await get_recommendations(request, req, db)