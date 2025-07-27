"""Logging utilities for CRISP."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Formats log records as JSON objects for easy parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


def setup_logger(
    name: str = "crisp",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_json: bool = True,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with specified configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        use_json: Whether to use JSON formatting
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter: Union[JSONFormatter, logging.Formatter]
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(config: Dict[str, Any], logger: logging.Logger):
    """
    Log configuration parameters.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration", extra={"config": config})


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Log evaluation metrics.

    Args:
        metrics: Dictionary of metrics
        step: Optional training step
        epoch: Optional epoch number
        logger: Logger instance (creates default if None)
    """
    if logger is None:
        logger = logging.getLogger("crisp.metrics")

    extra: Dict[str, Any] = {"metrics": metrics}
    if step is not None:
        extra["step"] = step
    if epoch is not None:
        extra["epoch"] = epoch

    logger.info("Metrics", extra=extra)


def log_training_step(
    loss: float,
    step: int,
    epoch: int,
    learning_rate: float,
    batch_size: int,
    logger: Optional[logging.Logger] = None,
):
    """
    Log training step information.

    Args:
        loss: Training loss
        step: Training step
        epoch: Epoch number
        learning_rate: Current learning rate
        batch_size: Batch size
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("crisp.training")

    logger.info(
        f"Step {step} - Loss: {loss:.4f}",
        extra={
            "loss": loss,
            "step": step,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
    )


def log_model_info(
    model_name: str,
    num_parameters: int,
    architecture: str,
    logger: Optional[logging.Logger] = None,
):
    """
    Log model information.

    Args:
        model_name: Model name
        num_parameters: Total number of parameters
        architecture: Model architecture description
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("crisp.model")

    logger.info(
        f"Model: {model_name}",
        extra={
            "model_name": model_name,
            "num_parameters": num_parameters,
            "architecture": architecture,
        },
    )


class TensorBoardLogger:
    """
    Simple TensorBoard logger wrapper.

    Provides a unified interface for logging to TensorBoard.
    """

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            self.writer = None
            self.enabled = False
            logging.warning(
                "TensorBoard not available. Install with: " "pip install tensorboard"
            )

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, values, step)

    def log_histogram(self, tag: str, values: Any, step: int):
        """Log a histogram."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()


# Create default logger
default_logger = setup_logger()


def get_logger(name: str = "crisp") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
