"""
Rate Limiting 中间件

防止恶意刷接口导致 API Key 被封或产生高额费用。

设计：
- RateLimiterBackend Protocol：限流后端接口，MVP 用内存实现，生产切换 Redis
- MemoryRateLimiter：基于内存的滑动窗口限流
- rate_limit：装饰器，便捷应用限流策略

限流策略（Constaint.md 定义）：
| 场景         | 限制           |
|-------------|---------------|
| 创建选题      | 每 IP 每分钟 5 次 |
| 用户确认      | 每 IP 每分钟 10 次 |
| 搜索知识库    | 每 IP 每分钟 20 次 |
"""
import time
from typing import Protocol
from functools import wraps

from fastapi import Request, HTTPException


class RateLimiterBackend(Protocol):
    """限流后端接口 — MVP 用内存实现，生产切换 Redis"""

    async def is_rate_limited(
        self, key: str, max_requests: int, window_seconds: int
    ) -> bool:
        """检查指定 key 是否超过频率限制，返回 True 表示超限"""
        ...

    async def get_remaining(
        self, key: str, max_requests: int, window_seconds: int
    ) -> int:
        """返回剩余可用请求数（用于设置 X-RateLimit-Remaining 头）"""
        ...


class MemoryRateLimiter:
    """
    基于内存的滑动窗口限流实现（MVP 默认）

    使用简单列表存储每个 key 的时间戳，
    每次检查时清理过期记录。
    """

    def __init__(self):
        self._requests: dict[str, list[float]] = {}

    async def is_rate_limited(
        self, key: str, max_requests: int, window_seconds: int
    ) -> bool:
        now = time.time()
        if key in self._requests:
            # 清理过期记录（滑动窗口）
            self._requests[key] = [
                t for t in self._requests[key]
                if now - t < window_seconds
            ]
        else:
            self._requests[key] = []

        if len(self._requests[key]) >= max_requests:
            return True
        self._requests[key].append(now)
        return False

    async def get_remaining(
        self, key: str, max_requests: int, window_seconds: int
    ) -> int:
        if key not in self._requests:
            return max_requests
        now = time.time()
        count = sum(1 for t in self._requests[key] if now - t < window_seconds)
        return max(0, max_requests - count)


# 全局限流器实例
rate_limiter: RateLimiterBackend = MemoryRateLimiter()


def rate_limit(max_requests: int = 5, window_seconds: int = 60):
    """
    限流装饰器

    用法：
        @app.post("/api/topics")
        @rate_limit(max_requests=5, window_seconds=60)
        async def create_topic(request: Request, ...):
            ...

    Args:
        max_requests: 窗口内最大请求数
        window_seconds: 时间窗口（秒）
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            key = f"rate_limit:{func.__name__}:{client_ip}"

            limited = await rate_limiter.is_rate_limited(
                key, max_requests, window_seconds
            )
            if limited:
                remaining = await rate_limiter.get_remaining(
                    key, max_requests, window_seconds
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"请求过于频繁，请 {window_seconds} 秒后重试",
                    headers={
                        "X-RateLimit-Remaining": str(remaining),
                        "Retry-After": str(window_seconds),
                    }
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator