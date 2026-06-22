"""
FastAPI 路由 — SSE 事件驱动架构

基于 asyncio.Queue 的事件总线，实现真正的实时推送：
- 工作流节点完成后调用 publish_event() 推送事件
- SSE event_generator 通过 await queue.get() 阻塞等待（零 CPU 消耗）
- 客户端断开时自动清理队列

Constaint.md 要求：
- 使用 SSE，禁止轮询
- 所有外部输入经过 sanitize_input() 清洗
- Rate Limiting 装饰器
"""
import asyncio
import json
import logging
import re
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from storage import db
from storage.repository import topic_repo
from services.workflow_service import workflow_service
from api.schemas import (
    CreateTopicRequest,
    ConfirmRequest,
    CreateTopicResponse,
    ConfirmResponse,
    TopicResponse,
    HealthResponse,
)
from api.middleware import rate_limit

logger = logging.getLogger(__name__)

# ============================================================================
# SSE 事件总线（Constaint.md 3.10 节）
# ============================================================================

# 每个 topic_id 对应一个 asyncio.Queue，工作流节点完成后直接推送事件
_event_queues: dict[int, asyncio.Queue] = {}
_event_queues_lock = asyncio.Lock()


async def get_or_create_queue(topic_id: int) -> asyncio.Queue:
    """获取或创建事件队列（线程安全）"""
    if topic_id not in _event_queues:
        async with _event_queues_lock:
            if topic_id not in _event_queues:
                _event_queues[topic_id] = asyncio.Queue(maxsize=100)
    return _event_queues[topic_id]


async def publish_event(topic_id: int, event: dict):
    """工作流节点完成后调用，推送事件到 SSE 客户端"""
    queue = await get_or_create_queue(topic_id)
    try:
        await queue.put(event)
    except asyncio.QueueFull:
        logger.warning(f"事件队列已满 topic_id={topic_id}，丢弃旧事件")
        # 丢弃最旧的事件，放入新事件
        try:
            queue.get_nowait()
            await queue.put(event)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


async def remove_queue(topic_id: int):
    """客户端断开时清理队列"""
    async with _event_queues_lock:
        _event_queues.pop(topic_id, None)
        logger.info(f"清理事件队列 topic_id={topic_id}")


# 注入 publish_event 回调到 WorkflowService
workflow_service._publish = publish_event

# ============================================================================
# 输入清洗（Constaint.md 安全约束）
# ============================================================================

MAX_INPUT_LENGTH = 4000
FORBIDDEN_PATTERNS = [
    r"\((?:system|user|assistant)\s*:\s*",
    r"\{\{.*\}\}",
    r"<\|.*\|>",
    r"<\/.*>",
]


def sanitize_input(value: str, field_name: str = "input") -> str:
    """
    对外部输入进行安全清洗

    防御：
    1. 长度截断
    2. 去除提示词攻击特征
    3. 去除多余空白
    """
    if not value:
        return ""

    # 1. 长度截断
    if len(value) > MAX_INPUT_LENGTH:
        value = value[:MAX_INPUT_LENGTH]
        logger.warning(f"sanitize: {field_name} 超过 {MAX_INPUT_LENGTH} 字符，已截断")

    # 2. 去除潜在提示词攻击特征
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            logger.warning(f"sanitize: {field_name} 包含可疑模式，已过滤")
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)

    # 3. 去除多余空白字符
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================================
# 路由定义
# ============================================================================


def create_routes(app: FastAPI | None = None) -> FastAPI:
    """创建并注册所有 API 路由到 FastAPI 应用"""
    if app is None:
        app = FastAPI(title="营销内容生成 Agent API")

    # --- SSE 事件流 ---
    @app.post("/api/topics/{topic_id}/events")
    async def sse_events(topic_id: int):
        """
        SSE 流式事件推送（真正的事件驱动，无轮询）

        客户端通过此接口实时获取工作流状态变更。
        底层使用 asyncio.Queue 阻塞等待，零 CPU 消耗。
        """
        topic = db.get_topic(topic_id)
        if not topic:
            raise HTTPException(status_code=404, detail="选题不存在")

        async def event_generator():
            queue = await get_or_create_queue(topic_id)

            try:
                # --- 排空队列中的待处理事件，避免使用过时的 DB 状态 ---
                # 场景：用户驳回后 st.rerun() 重新连接 SSE，此时后台工作流
                # 可能已经把新内容推送到了队列，但 DB 尚未更新或被已更新的
                # 队列事件覆盖。先排空队列，使用最新事件的数据作为初始状态。
                latest_pending: dict | None = None
                while not queue.empty():
                    try:
                        ev = queue.get_nowait()
                        # 找到最新的 waiting_confirm 或终态事件
                        if ev.get("status") in ("waiting_confirm", "completed", "failed"):
                            latest_pending = ev
                    except asyncio.QueueEmpty:
                        break

                if latest_pending is not None:
                    # 使用队列中最新的待确认/终态事件，而非过时的 DB 状态
                    logger.info(
                        f"SSE: 使用队列中的最新事件代替 DB 初始状态 "
                        f"topic_id={topic_id}"
                    )
                    yield f"data: {json.dumps(latest_pending, default=str)}\n\n"

                    if latest_pending.get("status") in ("completed", "failed"):
                        return
                else:
                    # 队列为空，使用 DB 中的状态
                    current_status = topic.get("status", "running")
                    current_step = topic.get("current_step", "planning")

                    initial_event: dict = {
                        'status': current_status,
                        'step': current_step,
                        'result': _get_step_result(topic, current_step),
                        'content': topic.get('content'),
                    }
                    # fail_reason 只在有实际错误描述时才包含，避免日志中
                    # 反复出现 'fail_reason': None 混淆视听
                    if topic.get('fail_reason'):
                        initial_event['fail_reason'] = topic['fail_reason']

                    yield f"data: {json.dumps(initial_event)}\n\n"

                    # 如果是终态，立即结束
                    if current_status in ("completed", "failed"):
                        return

                # 阻塞等待新事件
                while True:
                    event = await queue.get()  # 阻塞等待，零 CPU 消耗
                    yield f"data: {json.dumps(event, default=str)}\n\n"

                    # 终态时退出
                    if event.get("status") in ("completed", "failed"):
                        break

            except asyncio.CancelledError:
                pass  # 客户端断开
            finally:
                await remove_queue(topic_id)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- 创建选题 ---
    @app.post("/api/topics")
    @rate_limit(max_requests=5, window_seconds=60)
    async def create_topic(request: Request, req: CreateTopicRequest):
        """
        创建选题，启动工作流

        流程：
        1. 清洗用户输入（sanitize_input）
        2. 创建数据库记录
        3. 异步启动工作流
        4. 立即返回 topic_id
        """
        # 安全清洗输入
        clean_input = sanitize_input(req.user_input, "user_input")

        if not clean_input:
            raise HTTPException(status_code=400, detail="输入内容不能为空")

        # 创建数据库记录
        topic = db.create_topic(
            title=clean_input[:50],  # 标题取前50字符
            user_input=clean_input,
        )
        topic_id = topic["id"]

        logger.info(f"创建选题 topic_id={topic_id}, input_length={len(clean_input)}")

        # 异步启动工作流
        asyncio.create_task(
            workflow_service.start_workflow(topic_id, clean_input)
        )

        return CreateTopicResponse(topic_id=topic_id, status="running")

    # --- 获取选题状态 ---
    @app.get("/api/topics/{topic_id}")
    async def get_topic(topic_id: int):
        """获取选题及当前状态"""
        topic = db.get_topic(topic_id)
        if not topic:
            raise HTTPException(status_code=404, detail="选题不存在")
        return topic

    # --- 获取选题迭代历史 ---
    @app.get("/api/topics/{topic_id}/iterations")
    async def get_topic_iterations(topic_id: int):
        """获取选题的迭代历史"""
        topic = db.get_topic(topic_id)
        if not topic:
            raise HTTPException(status_code=404, detail="选题不存在")

        iterations = topic_repo.get_iteration_history(topic_id)
        return {"topic_id": topic_id, "iterations": iterations}

    # --- 用户确认 ---
    @app.post("/api/topics/{topic_id}/confirm")
    @rate_limit(max_requests=10, window_seconds=60)
    async def confirm_topic(
        request: Request, topic_id: int, req: ConfirmRequest
    ):
        """
        用户确认（采纳/不采纳+意见）

        工作流在用户确认后继续执行下一步。
        """
        topic = db.get_topic(topic_id)
        if not topic:
            raise HTTPException(status_code=404, detail="选题不存在")

        if topic["status"] != "waiting_confirm":
            raise HTTPException(
                status_code=400,
                detail=f"当前状态不允许确认: {topic['status']}",
            )

        # 安全清洗反馈
        clean_feedback = (
            sanitize_input(req.feedback, "feedback") if req.feedback else None
        )

        logger.info(
            f"用户确认 topic_id={topic_id}, "
            f"approved={req.approved}, step={topic['current_step']}"
        )

        # 异步继续工作流
        asyncio.create_task(
            workflow_service.continue_workflow(
                topic_id, req.approved, clean_feedback
            )
        )

        return ConfirmResponse(status="continued")

    # --- 健康检查 ---
    @app.get("/health")
    async def health_check():
        return HealthResponse(status="healthy")

    return app


# ============================================================================
# 辅助函数
# ============================================================================


def _get_step_result(topic: dict, step: str) -> str:
    """获取指定步骤的结果"""
    if step == "planning":
        return topic.get("plan", "") or ""
    elif step == "generating":
        return topic.get("content", "") or ""
    elif step == "reviewing":
        return topic.get("review_result", "") or ""
    elif step == "reflection":
        return topic.get("reflection", "") or ""
    return ""