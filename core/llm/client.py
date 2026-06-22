"""
LLM 客户端封装
支持 OpenAI SDK，兼容 DeepSeek 等 API
"""
from openai import OpenAI
from config import settings


def get_llm_client() -> OpenAI:
    """获取 LLM 客户端"""
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


def get_llm_model() -> str:
    """获取 LLM 模型名称"""
    return settings.LLM_MODEL
