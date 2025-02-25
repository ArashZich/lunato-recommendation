from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
import uuid
from datetime import datetime
import base64

from api.schemas.request import FaceAnalysisRequest
from api.schemas.response import FaceDetectionResponse, FaceAnalysisResponse
from api.utils.client_info import extract_client_info
from api.utils.validators import validate_image
from database.connection import get_database
from celery_app.tasks.face_detection import detect_face, preprocess_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=FaceDetectionResponse, tags=["Face Analysis"])
async def analyze_face(
    request: FaceAnalysisRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Analyze a face image to determine face shape.
    
    This endpoint:
    1. Validates the input image
    2. Detects the face in the image
    3. Analyzes the face shape
    4. Returns detection results immediately
    5. Processes full analysis and recommendations in the background
    """
    try:
        # Validate the image data
        if not validate_image(request.image):
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Generate user ID if not provided
        user_id = request.user_id or str(uuid.uuid4())
        
        # Extract client info if not provided
        client_info = request.client_info
        if not client_info:
            client_info = extract_client_info(req)
            
        # Create metadata
        metadata = {
            "client_info": client_info.dict() if client_info else {},
            "timestamp": datetime.utcnow().isoformat(),
            "request_ip": req.client.host
        }
            
        # Store the request in the database
        analysis_request = {
            "user_id": user_id,
            "request_time": datetime.utcnow(),
            "client_info": client_info.dict() if client_info else {},
            "status": "processing"
        }
        await db.analysis_requests.insert_one(analysis_request)
        
        # Process face detection in Celery task
        # For immediate response, start the detection task
        task = detect_face.delay(request.image, user_id, metadata)
        
        # Return an immediate response
        return FaceDetectionResponse(
            success=True,
            message="Face analysis started. Check status with the provided task ID.",
            detection_id=task.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing face analysis request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.get("/analyze/{task_id}", response_model=FaceAnalysisResponse, tags=["Face Analysis"])
async def get_analysis_result(
    task_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get the result of a face analysis task.
    
    This endpoint:
    1. Checks the status of a face analysis task
    2. Returns the results if available
    """
    try:
        # Check if the analysis result exists in the database
        analysis = await db.face_analyses.find_one({"detection_id": task_id})
        
        if not analysis:
            # Check if the detection exists but analysis is not complete
            detection = await db.face_detections.find_one({"_id": task_id})
            
            if not detection:
                raise HTTPException(status_code=404, detail="Analysis task not found")
                
            return FaceAnalysisResponse(
                success=True,
                message=f"Analysis in progress. Status: {detection.get('status', 'processing')}",
            )
        
        # Return the face shape analysis result
        return FaceAnalysisResponse(
            success=True,
            message="Face shape analysis completed",
            face_shape=analysis.get("face_shape"),
            confidence=analysis.get("confidence"),
            shape_metrics=analysis.get("shape_metrics")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis result: {str(e)}")


@router.post("/preprocess", tags=["Face Analysis"])
async def preprocess_face_image(
    request: FaceAnalysisRequest
):
    """
    Preprocess an image to improve face detection.
    
    This endpoint:
    1. Validates the input image
    2. Applies preprocessing to improve face detection
    3. Returns the preprocessed image
    """
    try:
        # Validate the image data
        if not validate_image(request.image):
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image preprocessing in Celery task
        result = preprocess_image.delay(request.image)
        processed_result = result.get(timeout=30)  # Wait for task to complete with 30s timeout
        
        if not processed_result.get("success"):
            raise HTTPException(status_code=500, detail=processed_result.get("message", "Preprocessing failed"))
            
        return {
            "success": True,
            "message": "Image preprocessed successfully",
            "preprocessed_image": processed_result.get("preprocessed_image")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")