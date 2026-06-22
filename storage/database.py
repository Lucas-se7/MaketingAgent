"""
数据库连接管理
支持 SQLite (MVP) 和 PostgreSQL (生产)
"""
import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from config import settings
from storage.models import Base

logger = logging.getLogger(__name__)


def create_db_engine():
    """创建数据库引擎"""
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.LOG_LEVEL == "DEBUG",
        # SQLite 特定配置
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
        poolclass=StaticPool if "sqlite" in settings.DATABASE_URL else None,
    )

    # SQLite 启用外键约束
    if "sqlite" in settings.DATABASE_URL:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine):
    """创建会话工厂"""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 全局引擎和会话工厂
engine = create_db_engine()
SessionLocal = create_session_factory(engine)


def _ensure_db_directory():
    """确保 SQLite 数据库文件的父目录存在"""
    if "sqlite" in settings.DATABASE_URL:
        # 从 sqlite:///./data/marketing.db 中提取路径部分
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"自动创建数据库目录: {db_dir}")


def init_db():
    """初始化数据库（创建目录和表）"""
    _ensure_db_directory()
    Base.metadata.create_all(bind=engine)
    logger.info("数据库初始化完成")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库会话错误: {e}")
        raise
    finally:
        session.close()


class DatabaseManager:
    """
    数据库管理器（MVP 简化版）

    提供基础的 CRUD 操作
    """

    def __init__(self):
        init_db()

    def create_topic(self, title: str, user_input: str) -> dict:
        """创建选题"""
        from storage.models import Topic

        with get_db_session() as session:
            topic = Topic(
                title=title,
                user_input=user_input,
                status="running",
                current_step="planning"
            )
            session.add(topic)
            session.commit()
            session.refresh(topic)
            return topic.to_dict()

    def get_topic(self, topic_id: int) -> dict | None:
        """获取选题"""
        from storage.models import Topic

        with get_db_session() as session:
            topic = session.query(Topic).filter(Topic.id == topic_id).first()
            return topic.to_dict() if topic else None

    def update_topic(self, topic_id: int, **kwargs) -> dict | None:
        """更新选题"""
        from storage.models import Topic

        with get_db_session() as session:
            topic = session.query(Topic).filter(Topic.id == topic_id).first()
            if not topic:
                return None

            for key, value in kwargs.items():
                if hasattr(topic, key):
                    setattr(topic, key, value)

            session.commit()
            session.refresh(topic)
            return topic.to_dict()

    def list_topics(self, limit: int = 20) -> list[dict]:
        """获取选题列表"""
        from storage.models import Topic

        with get_db_session() as session:
            topics = session.query(Topic).order_by(
                Topic.created_at.desc()
            ).limit(limit).all()
            return [t.to_dict() for t in topics]


# 全局数据库管理器
db = DatabaseManager()
