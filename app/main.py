from app.core.logging_config import setup_logging
setup_logging()
import asyncio
from asyncio import CancelledError
from fastapi import FastAPI
from app.database import init_db, close_db, SessionLocal, engine
from app.services.assist_service import AssistService
from app.controllers.assist_controller import router as assist_router
from app.controllers.auth_controller import router as auth_router
from app.controllers.health_controller import router as health_router
from app.controllers.prediction_controller import router as prediction_router
from app.controllers.token_credit_controller import router as token_credit_router
from app.controllers.train_model_controller import router as train_model_router
from app.controllers.user_controller import router as user_router
from app.controllers.user_usage_controller import router as user_usage_router
from app.utils.redis import init_redis, close_redis
from app.exceptions.handlers import app_exception_handlers
from app.maintenance.health import db_guard
from app.maintenance.reconciler import reconcile_trained_models_on_startup, reconcile_predictions_on_startup
import logging


app = FastAPI(title="FastAPI ML Project")

# Register routers
app.include_router(assist_router)
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(token_credit_router)
app.include_router(train_model_router)
app.include_router(user_router)
app.include_router(user_usage_router)


app_exception_handlers(app)


@app.on_event("startup")
async def on_startup():
    """Create tables on startup"""
    logging.info("Starting FastAPI ML Project")
    await init_db()
    await init_redis()

    async with SessionLocal() as db:
        await reconcile_trained_models_on_startup(db)
        await reconcile_predictions_on_startup(db)

    app.state.db_guard_task = asyncio.create_task(db_guard(engine))

    AssistService.init()


@app.on_event("shutdown")
async def on_shutdown():
    """Clean up DB engine"""
    logging.info("Shutting down FastAPI ML Project")

    t = getattr(app.state, "db_guard_task", None)
    if t:
        t.cancel()
        try:
            await t
        except CancelledError:
            pass

    await close_db()
    await close_redis()
    import logging as _logging
    _logging.shutdown()
