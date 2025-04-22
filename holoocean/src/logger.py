from loguru import logger
import sys 
from datetime import datetime
from pathlib import Path

POSE_LEVEL = 18  # Custom log level for PoseSensor messages
logger.level("POSE", no=POSE_LEVEL, color="<cyan>", icon="🛠️")

CAMERA_LEVEL = 19  # Custom log level for Camera messages
logger.level("CAMERA", no=CAMERA_LEVEL, color="<green>", icon="📷")

SONAR_LEVEL = 17  # Custom log level for Camera messages
logger.level("SONAR", no=SONAR_LEVEL, color="<green>", icon="📷")


def pose_filter(record):
    return  record["level"].no == POSE_LEVEL

def camera_filter(record):
    return  record["level"].no == CAMERA_LEVEL

def sonar_filter(record):
    return  record["level"].no == SONAR_LEVEL

def setup_logger(log_dir):
    """
    Set up the logger with the specified configuration.

    Args:
        log_dir (str): The directory where log files will be stored.
        name (str): The name of the logger.
        level (str): The logging level. Default is "DEBUG".
    """
    
    start_time = datetime.now()
    logger.remove()

    log_dir = Path(log_dir) / start_time.strftime("%Y-%m-%d_%H-%M-%S")

    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")

    logger.add(log_dir / "pose.log", filter=pose_filter, format="{message}", level="POSE")

    logger.add(log_dir / "images.log", filter=camera_filter, format="{message}", level="CAMERA")

    logger.add(log_dir / "sonar.log", filter=sonar_filter, format="{message}", level="SONAR")

    return log_dir 
