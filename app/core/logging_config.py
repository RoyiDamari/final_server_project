import os
import logging
from logging.config import dictConfig

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    os.makedirs("logs", exist_ok=True)

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
        },

        "handlers": {
            "errors_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/errors.log",
                "maxBytes": 10_000_000,
                "backupCount": 5,
                "formatter": "plain",
                "level": "ERROR",
            },
            "activity_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/activity.log",
                "maxBytes": 10_000_000,
                "backupCount": 5,
                "formatter": "plain",
                "level": "INFO",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "level": "ERROR",
            },
        },

        "loggers": {
            "app.errors": {"handlers": ["errors_file", "console"], "level": "INFO", "propagate": False},
            "app.activity": {"handlers": ["activity_file", "console"], "level": "INFO", "propagate": False},
        },
    })

    # 🔇 Silence SQLAlchemy noise
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)

    # Silence FastAPI access logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


errors = logging.getLogger("app.errors")
activity = logging.getLogger("app.activity")
