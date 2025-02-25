from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import logging
from api.config import get_settings

logger = logging.getLogger(__name__)

# Global database connection object
db = None
client = None


async def connect_to_database():
    """
    Creates a database connection.
    """
    global db, client
    
    settings = get_settings()
    
    try:
        logger.info(f"Connecting to MongoDB at {settings.mongodb_uri}")
        client = AsyncIOMotorClient(settings.mongodb_uri)
        
        # Verify the connection
        await client.admin.command('ping')
        
        db = client[settings.mongo_db_name]
        logger.info("Connected to MongoDB")
        
        # Create indexes for better performance
        await create_indexes()
        
    except ConnectionFailure as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise


async def create_indexes():
    """
    Creates indexes for better query performance
    """
    # User collection indexes
    await db.users.create_index("device_id", unique=True)
    
    # Face analysis collection indexes
    await db.face_analyses.create_index("user_id")
    await db.face_analyses.create_index("created_at")
    
    # Recommendations collection indexes
    await db.recommendations.create_index("user_id")
    await db.recommendations.create_index("face_analysis_id")


async def close_database_connection():
    """
    Closes the database connection.
    """
    global client
    
    if client:
        logger.info("Closing MongoDB connection")
        client.close()
        logger.info("MongoDB connection closed")


def get_database():
    """
    Returns the database instance
    """
    return db