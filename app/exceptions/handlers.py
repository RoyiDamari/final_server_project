import asyncio
from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.encoders import jsonable_encoder
from sqlalchemy.exc import IntegrityError, DataError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException
from .base import BaseAppException
from app.core.logs import level_for_status
from app.core.logging_config import errors
from app.config import config


def app_exception_handlers(app: FastAPI):
    """
    Registers custom exception handlers for the FastAPI app.

    Currently, handles:
    - BaseAppException: Logs structured error with IP, method, and URL,
        then returns a JSON response with the error details and appropriate status code.

    Skips logging for RateLimitException, InvalidCreditCardException, DeleteException to avoid log spam.

    Args:
        app (FastAPI): The FastAPI app instance.
    """
    def exc_name(exc, fq: bool = False) -> str:
        c = exc.__class__
        return f"{c.__module__}.{c.__name__}" if fq else c.__name__


    @app.exception_handler(BaseAppException)
    async def base_app_exception_handler(request: Request, exc: BaseAppException):
        headers = {}

        if hasattr(exc, "retry_after"):
            headers["Retry-After"] = str(exc.retry_after)

        if not getattr(exc, "suppress_log", False):
            if config.DEBUG and exc.__cause__ is not None:
                errors.exception(
                    f"{exc.__class__.__name__} at {request.method} {request.url.path}",
                    exc_info=exc,
                )
            else:
                errors.error(
                    f"{exc.__class__.__name__} at {request.method} {request.url.path} "
                    f"from {(request.client.host if request.client else 'unknown')}: "
                    f"{getattr(exc, 'log_detail', exc.detail)}"
                )

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )


    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(exc.errors())},
        )


    @app.exception_handler(ResponseValidationError)
    async def response_validation_handler(_: Request, exc: ResponseValidationError):
        errors.exception(
            f"Response validation failed: {exc_name(exc, fq=True)} {exc.errors()}"
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"detail": "Internal response validation error"})


    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger_fn = level_for_status(exc.status_code)
        logger_fn(
            f"{exc.status_code} {request.method} {request.url.path}"
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


    @app.exception_handler(DataError)
    async def sa_data_error_handler(_: Request, exc: DataError):
        errors.warning(f"DataError: {getattr(exc, 'orig', exc)}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": "Invalid data"})


    @app.exception_handler(OperationalError)
    async def sa_operational_error_handler(_: Request, exc: OperationalError):
        errors.error(
            f"OperationalError: {exc_name(exc, fq=True)} {getattr(exc, 'orig', exc)}"
        )
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": "Database unavailable"})


    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception):
        if isinstance(exc, asyncio.CancelledError):
            raise

        errors.exception(f"Unhandled exception: {exc_name(exc, fq=True)} {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"detail": "Unexpected application error. Please refresh or log in again."})
