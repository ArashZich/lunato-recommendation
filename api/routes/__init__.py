# Import routers to make them available via the api.routes namespace
from api.routes.analysis import router as analysis_router
from api.routes.recommendations import router as recommendations_router
from api.routes.analytics import router as analytics_router