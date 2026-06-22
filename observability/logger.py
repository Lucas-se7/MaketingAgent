"""
可观测性模块
包含日志配置
"""
from loguru import logger
import sys

from config import settings


def setup_logging():
    """配置日志"""
    # 移除默认 handler
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 添加文件输出
    logger.add(
        "logs/app.log",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger


# 初始化日志
setup_logging()
