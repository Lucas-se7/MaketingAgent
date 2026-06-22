"""
配置管理 - 环境变量校验
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """应用配置，含启动时校验"""

    # LLM 配置
    LLM_BASE_URL: str = Field(default="https://api.deepseek.com/v1")
    LLM_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="deepseek-chat")

    # 数据库配置
    DATABASE_URL: str = Field(default="sqlite:///./data/marketing.db")

    # 迭代控制
    MAX_ITERATIONS: int = Field(default=3, ge=1, le=10)

    # 日志
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("LLM_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("LLM_API_KEY 不能为空或过短，请检查配置")
        return v

    @field_validator("LLM_BASE_URL")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL 必须以 http:// 或 https:// 开头")
        return v.rstrip("/")

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("sqlite:///", "postgresql://")):
            raise ValueError("DATABASE_URL 必须以 sqlite:/// 或 postgresql:// 开头")
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 全局配置实例（启动时实例化即校验）
settings = Settings()
