# Import database models to make them available via the database.models namespace
from database.models.user import User
from database.models.face_analysis import FaceCoordinates, FaceDetection, FaceAnalysis
from database.models.recommendation import RecommendedFrame, Recommendation
from database.models.analytics import (
    AnalysisRequest, UserActivity, RecommendationAnalytics, AggregateStatistics
)