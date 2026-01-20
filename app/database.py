from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.config import config
from app.models.orm_models import Base

# Create one global engine
engine = create_async_engine(config.DATABASE_URL, echo=False, future=True)

# Create session factory
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Called at startup to create tables"""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Dependency for FastAPI routes"""
    async with SessionLocal() as session:
        async with session.begin():
            yield session

async def close_db():
    """Dispose the engine and release all connections."""
    await engine.dispose()
