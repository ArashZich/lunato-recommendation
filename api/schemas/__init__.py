# Import schema models to make them available via the api.schemas namespace
from api.schemas.request import ClientInfo, FaceAnalysisRequest, FrameRecommendationRequest, AnalyticsRequest
from api.schemas.response import (
    BaseResponse, FaceCoordinates, FaceDetectionResponse, 
    FaceAnalysisResponse, FrameRecommendationResponse, AnalyticsResponse,
    FaceShapeCount, FrameTypeCount, DeviceAnalytics, BrowserAnalytics
)
from api.schemas.analytics import (
    UserAnalytics, AggregateAnalytics, TimeSeriesData, TimeSeriesAnalytics,
    ConversionAnalytics, FaceShapeInsights, FrameTypeInsights
)