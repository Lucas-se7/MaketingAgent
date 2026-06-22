"""
LLM 调用追踪 - 可观测性支持
"""
import time
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)


def generate_with_tracing(
    client,
    messages: list[dict],
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    LLM 调用追踪包装器

    记录：
    - Prompt / Messages
    - Completion
    - Token 数量
    - 耗时
    """
    start_time = time.time()
    model = model or settings.LLM_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        duration = time.time() - start_time
        usage = response.usage

        logger.info(
            f"LLM Call | model={model} | "
            f"prompt_tokens={usage.prompt_tokens if usage else 0} | "
            f"completion_tokens={usage.completion_tokens if usage else 0} | "
            f"total_tokens={usage.total_tokens if usage else 0} | "
            f"duration={duration:.2f}s"
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"LLM Call FAILED | model={model} | "
            f"error={str(e)} | duration={duration:.2f}s"
        )
        raise


def generate_with_tools(
    client,
    messages: list[dict],
    tools: list[dict],
    model: Optional[str] = None,
    **kwargs
) -> tuple[str, list[dict]]:
    """
    LLM 调用（支持 Tool Calling）

    返回:
        (content, tool_calls)
    """
    start_time = time.time()
    model = model or settings.LLM_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        **kwargs
    )

    duration = time.time() - start_time
    usage = response.usage

    logger.info(
        f"LLM Call (tools) | model={model} | "
        f"tool_calls={len(response.choices[0].message.tool_calls or [])} | "
        f"total_tokens={usage.total_tokens if usage else 0} | "
        f"duration={duration:.2f}s"
    )

    message = response.choices[0].message
    content = message.content or ""
    tool_calls = [
        {"name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]

    return content, tool_calls
