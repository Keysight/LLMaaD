import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_log_action: ContextVar[str] = ContextVar("log_action", default="inference")


@contextmanager
def log_action(action: str):
    token = _log_action.set(action)
    try:
        yield
    finally:
        _log_action.reset(token)


def get_log_action() -> str:
    return _log_action.get()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prompt_reshaping")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
