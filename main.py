"""
营销内容生成 Agent - 主入口
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from storage import init_db
from api import create_routes

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 配置日志 — 同时输出到控制台和文件
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台
        logging.FileHandler("logs/app.log", encoding="utf-8"),  # 文件
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("启动营销内容生成 Agent...")
    logger.info(f"数据库: {settings.DATABASE_URL}")
    logger.info(f"LLM: {settings.LLM_BASE_URL}/{settings.LLM_MODEL}")

    # 初始化数据库
    init_db()

    yield

    # 关闭时
    logger.info("关闭营销内容生成 Agent...")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="营销内容生成 Agent",
        description="策划-生成-审核的营销内容自动生成工作流",
        version="0.1.0",
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    create_routes(app)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
