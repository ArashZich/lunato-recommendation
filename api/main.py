from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import analysis, recommendations, analytics
from api.config import Settings, get_settings
from database.connection import get_database, close_database_connection, connect_to_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Connect to the database when the application starts
    await connect_to_database()
    yield
    # Cleanup: Close database connection when the application shuts down
    await close_database_connection()


# Initialize FastAPI app
app = FastAPI(
    title="Eyeglass Frame Recommendation API",
    description="API for analyzing face shape and recommending eyeglass frames",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["Face Analysis"])
app.include_router(recommendations.router, prefix="/api/v1", tags=["Frame Recommendations"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])


@app.get("/", tags=["Health Check"])
async def root(settings: Settings = Depends(get_settings)):
    """
    Health check endpoint
    """
    return {
        "message": "Eyeglass Frame Recommendation API is running",
        "version": "1.0.0",
        "environment": settings.environment
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)