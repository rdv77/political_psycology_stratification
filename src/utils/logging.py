import logging
import logging.handlers
import os
from typing import Optional


def configure_logging(config: dict, quiet_flag: bool = False) -> None:
    logging_config = config.get("logging", {})

    mode = logging_config.get("mode", "dev")
    if quiet_flag:
        mode = "quiet"

    console_enabled = logging_config.get("console_enabled", True)
    file_enabled = logging_config.get("file_enabled", True)
    console_level = logging_config.get("console_level", "DEBUG")
    file_level = logging_config.get("file_level", "INFO")
    fmt = logging_config.get("format", "text")

    if mode == "prod":
        console_enabled = False
        file_level = "CRITICAL"
    if mode == "quiet":
        console_enabled = False
        file_level = "CRITICAL"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = _make_formatter(fmt)

    if console_enabled:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level, logging.DEBUG))
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if file_enabled:
        file_path = logging_config.get("file_path", "logs/app.log")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        max_bytes = int(logging_config.get("rotate_max_bytes", 10 * 1024 * 1024))
        backups = int(logging_config.get("rotate_backup_count", 5))
        fh = logging.handlers.RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        fh.setLevel(getattr(logging, file_level, logging.INFO))
        fh.setFormatter(formatter)
        root.addHandler(fh)


def _make_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        try:
            import orjson

            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    payload = {
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        payload["exc_info"] = self.formatException(record.exc_info)
                    return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8")

            return JsonFormatter()
        except Exception:
            pass
    # fallback to text
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or __name__)


