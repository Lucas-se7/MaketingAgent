# 营销内容自动生成 Agent - 技术架构

## 1. 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **工作流** | LangGraph | 多步骤工作流编排，内置状态管理 |
| **LLM** | OpenAI SDK | 轻量调用，兼容 DeepSeek 等 API |
| **数据库** | SQLite / PostgreSQL | MVP 用 SQLite，生产切换 PostgreSQL |
| **ORM** | SQLAlchemy | 数据模型管理，支持多数据库 |
| **前端** | Streamlit | MVP 快速验证 |
| **配置** | Pydantic Settings | 类型安全的环境变量 |
| **日志** | Loguru | 简化日志 |
| **可观测性** | LangSmith / OpenTelemetry | LLM 调用追踪（Prompt/Completion/Token） |
| **包管理** | uv | 高速依赖管理 |

**LLM 配置**（通过 `.env`）：
```bash
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-key
LLM_MODEL=deepseek-chat
```

**数据库配置**：
```bash
# MVP - SQLite（轻量，无需服务）
DATABASE_URL=sqlite:///./data/marketing.db

# 生产 - PostgreSQL（并发支持）
DATABASE_URL=postgresql://user:pass@localhost:5432/marketing
```

**SQLite 注意事项**：应用层需做好连接池管理（SQLAlchemy 内置），写入操作使用 `BEGIN IMMEDIATE` 事务避免锁竞争。

---

## 2. 目录结构

```
marketing_agent/
├── core/
│   ├── agent/
│   │   ├── state.py       # 工作流状态（分状态类）
│   │   ├── nodes.py       # 节点：planner / generator / reviewer / reflection
│   │   ├── graph.py       # LangGraph 工作流编排
│   │   └── prompts.py      # Prompt 管理（从数据库加载）
│   ├── llm/
│   │   ├── client.py      # OpenAI SDK 封装
│   │   └── tracing.py     # LLM 调用追踪
│   └── tools/
│       └── sensitive.py    # 敏感词检测
├── storage/
│   ├── models.py          # SQLAlchemy 模型
│   ├── database.py        # 数据库连接
│   └── repository.py      # 状态序列化/反序列化
├── api/
│   ├── routes.py          # API 路由
│   ├── schemas.py         # Pydantic 模型
│   └── middleware.py      # Rate Limiting 中间件
├── ui/
│   └── streamlit_app.py   # Streamlit 前端（轮询模式）
├── services/              # 业务逻辑
├── config/
│   └── settings.py        # 配置管理（含环境变量校验）
├── observability/
│   └── logger.py          # 可观测性配置
├── main.py                # 应用入口（FastAPI）
└── .env
```

---

## 3. 核心设计

### 3.1 工作流架构

基于 LangGraph 的工作流，支持反思闭环和主动工具调用：

```
START → [策划] → 等待确认 → [生成] → 等待确认 → [审核] → END
                   ↓采纳                      ↓采纳
                   ↓不采纳则重新策划           ↓不采纳
                                          [反思] → 更新Plan → [生成]
                                                            ↑用户确认
```

**核心改进：**
- **ReflectionNode**：审核驳回后，Agent 分析原因并更新策划方案，实现真正的闭环学习
- **Tool Calling**：Agent 可主动调用工具（搜索网络/查询数据库/检索知识库），而非被动接收

### 3.2 状态定义（Pydantic State 模型）

使用 Pydantic 替代 dataclass，利用其序列化能力，降低维护成本：

```python
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# --- 分状态类（Pydantic Model）---

class PlanState(BaseModel):
    plan: str | None = None
    approved: bool | None = None
    feedback: str | None = None

class ContentState(BaseModel):
    content: str | None = None
    approved: bool | None = None
    feedback: str | None = None

class ReviewState(BaseModel):
    result: str | None = None          # 通过/驳回 + 理由
    approved: bool | None = None

class ReflectionState(BaseModel):
    analysis: str | None = None        # 反思分析结果
    updated_plan: str | None = None   # 更新后的策划方案
    previous_rejected_plan: str | None = None  # 上一轮被驳回的 plan，用于收敛检测
    iteration_count: int = 0

class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict
    result: any
    timestamp: str

# --- 主状态类 ---

class AgentState(BaseModel):
    # 用户输入
    user_input: str = ""

    # 分状态
    plan_state: PlanState = Field(default_factory=PlanState)
    content_state: ContentState = Field(default_factory=ContentState)
    review_state: ReviewState = Field(default_factory=ReviewState)
    reflection_state: ReflectionState = Field(default_factory=ReflectionState)

    # 工具调用记录
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # 当前步骤
    step: str = "planning"

    # 迭代控制（防止无限循环）
    max_iterations: int = 3
    current_iteration: int = 0

    # 收敛检测（防止反思无效循环）
    convergence_threshold: float = 0.9  # 连续两次 plan 相似度 > 0.9 认为已收敛

    # 状态标记
    status: str = "running"  # running / waiting_confirm / completed / failed
    fail_reason: str | None = None

    # 用户确认超时
    waiting_since: datetime | None = None  # 进入 waiting_confirm 的时间戳
    confirmation_timeout_minutes: int = 30 # 确认超时时间

    def get_current_result(self) -> str:
        """获取当前步骤的结果"""
        if self.step == "planning":
            return self.plan_state.plan or ""
        elif self.step == "generating":
            return self.content_state.content or ""
        elif self.step == "reviewing":
            return self.review_state.result or ""
        return ""

    def is_iteration_exceeded(self) -> bool:
        """检查是否超过最大迭代次数"""
        return self.current_iteration >= self.max_iterations

    def is_confirmation_timed_out(self) -> bool:
        """检查用户确认是否超时"""
        if self.waiting_since is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.waiting_since).total_seconds()
        return elapsed > self.confirmation_timeout_minutes * 60

    def is_reflection_converged(self) -> bool:
        """检查反思是否已收敛（连续两次 updated_plan 高度相似）"""
        current = self.reflection_state.updated_plan
        previous = self.reflection_state.previous_rejected_plan
        if not current or not previous:
            return False
        similarity = SequenceMatcher(None, current, previous).ratio()
        return similarity >= self.convergence_threshold

    class Config:
        arbitrary_types_allowed = True  # 允许 dict 类型字段
```

**Pydantic 优势**：
- 内置序列化/反序列化（`.model_dump_json()`）
- 自动校验类型
- 兼容 LangGraph 的 `TypedDict` 用法

### 3.3 节点设计

| 节点 | 输入 | 输出 | 工具调用 |
|------|------|------|----------|
| **PlannerNode** | 用户选题 | 策划方案（受众/方向/卖点/风格） | 可调用知识库搜索 |
| **GeneratorNode** | 策划方案 + 模板 + 风格 | 营销文案（生成后即过滤） | 模板获取 + **敏感词检测** |
| **ReviewerNode** | 内容 + 知识库 | 审核结果（通过/驳回 + 理由） | 敏感词检测 |
| **ReflectionNode** | 驳回理由 + 审核结果 | 反思分析 + 更新后的策划方案 | 可调用网络搜索/知识库补充 |

**双重敏感词拦截**：
- **GeneratorNode 生成后**：首次敏感词过滤，防止 LLM 直接产生违规内容
- **ReviewerNode 审核时**：二次敏感词检测，双重保障

### 3.4 Tool Calling 设计

Agent 可主动调用工具（OpenAI Function Calling）：

```python
# 工具定义
tools = [
    {
        "name": "search_knowledge",
        "description": "搜索品牌知识库",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            }
        }
    },
    {
        "name": "search_web",
        "description": "搜索网络获取最新信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索内容"}
            }
        }
    },
    {
        "name": "get_template",
        "description": "获取内容模板",
        "parameters": {
            "type": "object",
            "properties": {
                "style": {"type": "string", "description": "风格类型"}
            }
        }
    },
    {
        "name": "check_sensitive",
        "description": "敏感词检测",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "待检测内容"}
            }
        }
    }
]
```

### 3.5 LLM 可观测性

集成 LangSmith 或 OpenTelemetry，记录每一次 LLM 调用：

```python
# core/llm/tracing.py
from langsmith import traceable
from opentelemetry import trace

@traceable(name="llm.generate", tags=["llm"])
def generate_with_tracing(client, prompt: str, **kwargs):
    """LLM 调用追踪包装器"""
    import time
    start_time = time.time()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )

    # 记录指标（结构化日志，便于接入可观测平台）
    duration = time.time() - start_time
    usage = response.usage

    logger.bind(
        model=settings.LLM_MODEL,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        duration=f"{duration:.2f}s",
    ).info("LLM call completed")

    return response.choices[0].message.content
```

**追踪内容**：
| 指标 | 说明 |
|------|------|
| Prompt | 发送给 LLM 的完整 prompt |
| Completion | LLM 返回的完整响应 |
| Token 数量 | prompt_tokens / completion_tokens / total_tokens |
| 耗时 | 请求到响应的延迟 |
| 错误 | 异常信息（如有） |

**配置**：
```bash
# .env
LANGSMITH_API_KEY=your-langsmith-key  # 可选
LANGSMITH_PROJECT=marketing-agent      # 项目名
OTEL_EXPORTER=console                  # 或 jaeger/otlp
```

### 3.6 ReflectionNode 反思逻辑

审核驳回后，Agent 分析原因并更新策划方案，带迭代次数保护：

```python
def reflection_node(state: AgentState) -> AgentState:
    """反思节点：分析驳回原因，更新策划方案"""
    # 检查迭代次数，防止无限循环
    if state.is_iteration_exceeded():
        raise MaxIterationsExceededError(
            f"已达到最大迭代次数 ({state.max_iterations})，"
            f"请人工介入或调整策划方案"
        )

    rejection_reason = state.review_state.result
    feedback = state.content_state.feedback

    analysis_prompt = f"""
    审核驳回原因：{rejection_reason}
    用户反馈：{feedback}

    请分析：
    1. 驳回的核心问题是什么？
    2. 需要补充哪些信息？
    3. 如何调整策划方案？
    """

    # Agent 可主动调用 tools 补充信息
    reflection = agent.run(analysis_prompt, tools=tools)

    # 保存上一轮被驳回的 plan，用于收敛检测
    state.reflection_state.previous_rejected_plan = \
        state.reflection_state.updated_plan or state.plan_state.plan or ""

    state.reflection_state.analysis = reflection.analysis
    state.reflection_state.updated_plan = reflection.new_plan
    state.reflection_state.iteration_count += 1
    state.current_iteration += 1

    # 收敛检测：连续两次反思结果高度相似 → 提前终止
    if state.is_reflection_converged():
        state.status = "failed"
        state.fail_reason = (
            f"反思已收敛，连续两次更新的策划方案高度相似"
            f"（相似度 ≥ {state.convergence_threshold}），"
            f"请人工介入调整方向"
        )
        raise ReflectionConvergedError(state.fail_reason)

    return state


class MaxIterationsExceededError(Exception):
    """超过最大迭代次数异常"""
    pass


class ReflectionConvergedError(Exception):
    """反思收敛异常（连续两次更新方向相似，提前终止）"""
    pass
```

**迭代控制流程**：
```
生成内容 → 审核通过? → 是 → END
              ↓否
         检查迭代次数
              ↓
    未超限 → 反思 → 收敛检测
              ↓          ↓
         更新Plan   已收敛 → 标记失败，人工介入
              ↓
         重新生成
              ↓
    已超限 → 抛出异常 → 人工介入
```

### 3.7 熔断机制（Graph 层）

在 Graph 的循环逻辑中检查迭代次数，防止死循环：

```python
# core/agent/graph.py
from langgraph.graph import StateGraph, END

def create_workflow():
    workflow = StateGraph(AgentState)

    # 添加节点...
    workflow.add_node("planner", planner_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("reflection", reflection_node)

    # 设置边
    workflow.add_edge("planner", "user_confirm")
    workflow.add_edge("generator", "user_confirm")
    workflow.add_edge("reviewer", "user_confirm")
    workflow.add_edge("reflection", "generator")  # 反思后重新生成

    # 条件边：审核结果决定下一步
    def should_continue(state: AgentState) -> str:
        if state.review_state.approved:
            return END
        if state.is_iteration_exceeded():
            return "fail"  # 超过迭代次数，标记失败
        if state.is_reflection_converged():
            return "fail"  # 反思收敛，提前终止
        return "reflection"

    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "reflection": "reflection",
            END: END,
            "fail": END,  # 熔断，直接结束
        }
    )

    # 条件边：用户确认超时处理
    def handle_confirmation(state: AgentState) -> str:
        if state.is_confirmation_timed_out():
            state.status = "failed"
            state.fail_reason = f"用户确认超时（{state.confirmation_timeout_minutes} 分钟）"
            return "fail"
        if state.plan_state.approved or state.content_state.approved:
            return "continue"
        return "wait_confirm"

    workflow.add_conditional_edges(
        "user_confirm",
        handle_confirmation,
        {
            "continue": "generator",
            "wait_confirm": "user_confirm",
            "fail": END,
        }
    )

    return workflow.compile()
```

**状态标记**：
```python
@dataclass
class AgentState:
    # ... 其他字段
    status: str = "running"  # running / waiting_confirm / completed / failed
    fail_reason: str | None = None  # 失败原因

    # 用户确认超时相关字段（详见 3.2 节完整定义）
    waiting_since: datetime | None = None
    confirmation_timeout_minutes: int = 30
```

**熔断规则总览**：
| 场景 | 触发条件 | 结果 |
|------|---------|------|
| 迭代超限 | `current_iteration >= max_iterations` | `status=failed`，人工介入 |
| 反思收敛 | 连续两次 `updated_plan` 相似度 ≥ 0.9 | `status=failed`，人工介入 |
| 确认超时 | `waiting_confirm` 超过 30 分钟无响应 | `status=failed`，释放资源 |

**超时处理流程**：
```
进入 waiting_confirm → 记录 waiting_since 时间戳
                          ↓
                  每次条件边检查 is_confirmation_timed_out()
                          ↓
              未超时 → 继续等待
              已超时 → 标记 failed → END
```

### 3.8 Tool Calling 错误处理

每个节点调用工具时使用 try-except，避免崩溃：

```python
# core/agent/nodes.py
import traceback
from typing import Any

def call_tool_safely(tool_name: str, args: dict, default_on_error: Any = None) -> dict:
    """安全调用工具，失败时返回错误信息而非崩溃"""
    try:
        tool = get_tool(tool_name)
        result = tool.invoke(args)
        return {"success": True, "result": result}
    except NetworkError as e:
        logger.warning(f"网络搜索失败: {e}")
        return {"success": False, "error": f"搜索失败：网络错误 - {e}"}
    except Exception as e:
        logger.error(f"工具调用失败: {tool_name}, error={e}")
        return {"success": False, "error": f"工具调用失败: {e}"}

def planner_node(state: AgentState) -> AgentState:
    """策划节点"""
    prompt = f"用户选题：{state.user_input}\n请制定策划方案..."

    # LLM 生成（可能调用工具）
    response = llm.generate(prompt, tools=available_tools)

    # 处理工具调用结果
    if response.tool_calls:
        for call in response.tool_calls:
            tool_result = call_tool_safely(call.name, call.args)
            if not tool_result["success"]:
                # 通知 Agent 工具失败，继续执行
                response.content += f"\n\n[系统提示] {tool_result['error']}"

    state.plan_state.plan = response.content
    return state
```

### 3.9 环境变量校验

启动时校验关键配置，避免运行时才发现问题：

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    # LLM 配置
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str = "deepseek-chat"

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./data/marketing.db"

    # 迭代控制
    MAX_ITERATIONS: int = 3

    @field_validator("LLM_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("LLM_API_KEY 不能为空或过短")
        return v

    @field_validator("LLM_BASE_URL")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL 必须以 http:// 或 https:// 开头")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略额外字段
```

**启动时自动校验**：
```python
# main.py
from config.settings import Settings

settings = Settings()  # 如果校验失败，应用启动时就报错
print(f"Loaded config: model={settings.LLM_MODEL}")
```

### 3.10 前端交互（SSE 流式输出）

升级为 SSE (Server-Sent Events)，替代轮询机制，实现流式输出，提供更丝滑的用户体验：

**架构**：
```
Streamlit UI  ←→  FastAPI (SSE)  ←→  Event Bus (asyncio.Queue)
                                            ↕
                                     LangGraph Agent
                                            ↕
                                        SQLite
```

**事件总线原理**：
- 每个 `topic_id` 对应一个 `asyncio.Queue`
- 工作流节点完成后调用 `publish_event()` 推送事件
- SSE `event_generator` 通过 `await queue.get()` 阻塞等待（零 CPU 消耗）
- 客户端断开时自动清理队列

**FastAPI SSE 后端**（基于 `asyncio.Queue` 事件总线，真正的事件驱动）：

```python
# api/routes.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
from collections import defaultdict

app = FastAPI()

# === 事件总线 ===
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
    await queue.put(event)

async def remove_queue(topic_id: int):
    """客户端断开时清理队列"""
    async with _event_queues_lock:
        _event_queues.pop(topic_id, None)

@app.post("/api/topics/{topic_id}/events")
async def sse_events(topic_id: int):
    """SSE 流式事件推送（真正的事件驱动，无轮询）"""
    async def event_generator():
        queue = await get_or_create_queue(topic_id)

        try:
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
            "X-Accel-Buffering": "no"
        }
    )
```

**工作流节点中发布事件**：

```python
# core/agent/nodes.py — 每个节点完成后推送事件
async def planner_node(state: AgentState) -> AgentState:
    # ... 策划逻辑
    state.plan_state.plan = response.content
    state.status = "waiting_confirm"
    state.waiting_since = datetime.now(timezone.utc)

    # 推送事件到 SSE 客户端
    await publish_event(state.topic_id, {
        "status": state.status,
        "step": "planning",
        "result": state.plan_state.plan,
    })
    return state

async def generator_node(state: AgentState) -> AgentState:
    # ... 生成逻辑
    # 推送实时进度
    await publish_event(state.topic_id, {
        "status": "running",
        "step": "generating",
        "result": "[生成中...]",
    })

    state.content_state.content = response.content
    state.status = "waiting_confirm"
    state.waiting_since = datetime.now(timezone.utc)

    await publish_event(state.topic_id, {
        "status": state.status,
        "step": "generating",
        "result": state.content_state.content,
    })
    return state
```

**Streamlit 前端**（SSE 监听）：
```python
# ui/streamlit_app.py
import streamlit as st
import requests
import sseclient  # pip install sseclient-py

st.set_page_config(page_title="营销内容生成")

def listen_sse(topic_id: int):
    """监听 SSE 事件，实时更新 UI"""
    response = requests.post(
        f"http://localhost:8000/api/topics/{topic_id}/events",
        stream=True
    )

    client = sseclient.SSEClient(response)
    for event in client.events():
        data = json.loads(event.data)

        if data["status"] == "running":
            st.info(f"⏳ {data['step'].upper()} 处理中...")
            if data.get("result"):
                st.text_area("当前结果", value=data["result"], height=150, disabled=True)

        elif data["status"] == "waiting_confirm":
            st.subheader("📋 请确认当前结果")
            st.text_area("结果", value=data.get("result", ""), height=200, disabled=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 通过"):
                    requests.post(f"/api/topics/{topic_id}/confirm",
                                  json={"approved": True})
            with col2:
                feedback = st.text_input("驳回理由", key="feedback_input")
                if st.button("❌ 驳回", disabled=not feedback):
                    requests.post(f"/api/topics/{topic_id}/confirm",
                                  json={"approved": False, "feedback": feedback})

        elif data["status"] == "completed":
            st.success("🎉 内容生成完成！")
            st.text_area("最终内容", value=data.get("content", ""), height=300)
            break

        else:  # failed
            st.error(f"❌ 生成失败: {data.get('fail_reason', '未知错误')}")
            break

def main():
    st.title("营销内容生成 Agent")

    user_input = st.text_input("请输入选题", placeholder="例如：新品发布...")

    if st.button("🚀 开始生成"):
        resp = requests.post(
            "http://localhost:8000/api/topics",
            json={"user_input": user_input}
        )
        topic_id = resp.json()["topic_id"]
        st.session_state["topic_id"] = topic_id

    # 监听 SSE 事件
    if "topic_id" in st.session_state:
        listen_sse(st.session_state["topic_id"])
```

**SSE vs 轮询（隐式 vs 真正事件驱动）**：
| 对比 | 轮询 (HTTP) | SSE + 隐式轮询 (`sleep`) | SSE + 事件总线 (`asyncio.Queue`) |
|------|------------|------------------------|-------------------------------|
| 延迟 | 1-2 秒 | 0-1 秒 | **实时** |
| 请求数 | 多次 HTTP | 单次长连接 | 单次长连接 |
| 服务器 CPU | 高（频繁建连） | 中（空转查询） | **低**（阻塞等待） |
| 实现复杂度 | 低 | 中 | 中 |
| 适用场景 | 原型 | MVP 初期 | **推荐** |

### 3.11 数据模型与状态持久化

**数据库 Schema**：
```sql
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    user_input TEXT NOT NULL,

    -- 策划阶段
    plan TEXT,
    plan_approved BOOLEAN,
    plan_feedback TEXT,

    -- 生成阶段
    content TEXT,
    content_approved BOOLEAN,
    content_feedback TEXT,

    -- 审核阶段
    review_result TEXT,
    review_approved BOOLEAN,

    -- 反思阶段
    reflection TEXT,
    updated_plan TEXT,
    iteration_count INTEGER DEFAULT 0,

    -- 元数据
    status TEXT DEFAULT 'running',     -- running / waiting_confirm / completed / failed
    fail_reason TEXT,
    current_step TEXT DEFAULT 'planning',
    waiting_since TIMESTAMP,           -- 进入 waiting_confirm 的时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 迭代历史表（记录每次迭代的完整快照，支持审计和复盘）
CREATE TABLE topic_iterations (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    iteration_num INTEGER NOT NULL,

    -- 该轮迭代的输入输出快照
    plan_snapshot TEXT,               -- 当轮策划方案
    content_snapshot TEXT,            -- 当轮生成内容
    review_result TEXT,               -- 审核结果
    reflection_analysis TEXT,         -- 反思分析（如果有）

    -- 该轮元数据
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / approved / rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_iterations_topic ON topic_iterations(topic_id, iteration_num);

CREATE TABLE templates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,      -- 含 {变量} 占位符
    style TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_items (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,  -- planner / generator / reviewer / reflection
    content TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**状态序列化/反序列化**：
```python
# storage/repository.py

class TopicRepository:
    def save(self, topic: Topic) -> None:
        """将 Topic 对象逐字段序列化到数据库（参数化查询，避免 SQL 注入）"""
        db.execute(
            """
            UPDATE topics SET
                title = ?, user_input = ?,
                plan = ?, plan_approved = ?, plan_feedback = ?,
                content = ?, content_approved = ?, content_feedback = ?,
                review_result = ?, review_approved = ?,
                reflection = ?, updated_plan = ?, iteration_count = ?,
                status = ?, fail_reason = ?, current_step = ?,
                waiting_since = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            [
                topic.title, topic.user_input,
                topic.plan_state.plan, topic.plan_state.approved, topic.plan_state.feedback,
                topic.content_state.content, topic.content_state.approved, topic.content_state.feedback,
                topic.review_state.result, topic.review_state.approved,
                topic.reflection_state.analysis, topic.reflection_state.updated_plan,
                topic.reflection_state.iteration_count,
                topic.status, topic.fail_reason, topic.step,
                topic.waiting_since, topic.id,
            ]
        )

    def save_iteration_snapshot(self, topic: Topic) -> None:
        """保存当前迭代的完整快照到 topic_iterations 表"""
        db.execute(
            """
            INSERT INTO topic_iterations
                (topic_id, iteration_num, plan_snapshot, content_snapshot,
                 review_result, reflection_analysis, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                topic.id,
                topic.current_iteration,
                topic.plan_state.plan,
                topic.content_state.content,
                topic.review_state.result,
                topic.reflection_state.analysis,
                "approved" if topic.review_state.approved else "rejected",
            ]
        )

    def load(self, topic_id: int) -> Topic:
        """从数据库记录反序列化为 Topic 对象"""
        row = db.query("SELECT * FROM topics WHERE id = ?", topic_id)
        topic = Topic(id=row["id"])
        topic.plan_state = PlanState(
            plan=row["plan"], approved=row["plan_approved"],
            feedback=row["plan_feedback"]
        )
        topic.content_state = ContentState(
            content=row["content"], approved=row["content_approved"],
            feedback=row["content_feedback"]
        )
        topic.review_state = ReviewState(
            result=row["review_result"], approved=row["review_approved"]
        )
        topic.reflection_state = ReflectionState(
            analysis=row["reflection"], updated_plan=row["updated_plan"],
            iteration_count=row["iteration_count"] or 0
        )
        topic.status = row["status"]
        topic.fail_reason = row["fail_reason"]
        topic.step = row["current_step"]
        topic.waiting_since = row["waiting_since"]
        return topic

    def get_iteration_history(self, topic_id: int) -> list[dict]:
        """获取指定选题的所有迭代历史（按时间排序）"""
        rows = db.query(
            "SELECT * FROM topic_iterations WHERE topic_id = ? ORDER BY iteration_num",
            topic_id
        )
        return [dict(row) for row in rows]
```

### 3.12 Prompt 治理规范

将 Prompt 从代码剥离，存入数据库或配置文件，便于调整 AI 行为而不改代码：

```sql
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,  -- planner / generator / reviewer / reflection
    content TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Prompt 加载**：
```python
# core/agent/prompts.py
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    content: str
    version: int

class PromptManager:
    def __init__(self, db):
        self.db = db
        self._cache: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate:
        """获取 Prompt，支持缓存"""
        if name not in self._cache:
            row = self.db.query(
                "SELECT * FROM prompts WHERE name = ?", name
            )
            self._cache[name] = PromptTemplate(**row)
        return self._cache[name]

    def update(self, name: str, content: str):
        """更新 Prompt（不改代码就能调整 AI 行为）"""
        self.db.execute(
            "UPDATE prompts SET content = ?, version = version + 1 WHERE name = ?",
            content, name
        )
        self._cache.pop(name, None)  # 清除缓存
```

**Prompt 模板示例**：
```sql
-- 策划节点 Prompt
INSERT INTO prompts (name, content) VALUES ('planner', '
你是一个营销策划专家。用户要推广：{user_input}

请制定策划方案，包含：
1. 目标受众
2. 内容方向
3. 核心卖点
4. 建议风格（专业/活泼/感性/幽默）

只输出策划方案，不要多余的话。
');

-- 生成节点 Prompt
INSERT INTO prompts (name, content) VALUES ('generator', '
你是营销文案专家。根据以下策划方案生成内容：

策划方案：
{plan}

风格：{style}
模板：{template}

要求：
1. 突出核心卖点
2. 语言简洁有力
3. 符合平台调性
');
```

### 3.13 Prompt 治理 - Few-shot Examples 管理

在 Prompt 基础上增加示例管理，对控制 LLM 输出格式至关重要：

```sql
CREATE TABLE prompt_examples (
    id INTEGER PRIMARY KEY,
    prompt_name TEXT NOT NULL,   -- 关联的 prompt (planner / generator / reviewer)
    example_input TEXT NOT NULL,  -- 示例输入
    example_output TEXT NOT NULL, -- 示例输出
    sort_order INTEGER DEFAULT 0, -- 排序顺序
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Examples 加载**：
```python
# core/agent/prompts.py
import re
from typing import Any

# --- 注入内容安全配置 ---
MAX_INPUT_LENGTH = 4000      # 单个变量最大长度
MAX_PROMPT_LENGTH = 8000    # 最终 Prompt 最大长度
FORBIDDEN_PATTERNS = [       # 禁止的模式（提示词攻击特征）
    r"\(system|user|assistant)\s*:\s*",
    r"\{\{.*\}\}",
    r"<\|.*\|>",
    r"<\/.*>",
]

class PromptManager:
    def __init__(self, db):
        self.db = db
        self._cache: dict[str, PromptTemplate] = {}
        self._examples_cache: dict[str, list] = {}

    def sanitize_input(self, value: str, field_name: str) -> str:
        """对注入到 Prompt 的外部内容进行清洗和截断"""
        if not value:
            return ""

        # 1. 长度截断
        if len(value) > MAX_INPUT_LENGTH:
            value = value[:MAX_INPUT_LENGTH]
            logger.warning(f"输入 {field_name} 超过 {MAX_INPUT_LENGTH} 字符，已截断")

        # 2. 去除潜在提示词攻击特征
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"输入 {field_name} 包含可疑模式 {pattern}，已过滤")
                value = re.sub(pattern, "", value, flags=re.IGNORECASE)

        # 3. 去除多余空白字符
        value = re.sub(r"\s+", " ", value).strip()

        return value

    def get_with_examples(self, name: str) -> tuple[str, list[dict]]:
        """获取 Prompt 及关联的 Few-shot Examples"""
        prompt = self.get(name)

        if name not in self._examples_cache:
            rows = self.db.query(
                "SELECT * FROM prompt_examples WHERE prompt_name = ? ORDER BY sort_order",
                name
            )
            self._examples_cache[name] = [
                {"input": r.example_input, "output": r.example_output}
                for r in rows
            ]

        return prompt.content, self._examples_cache[name]

    def format_prompt(self, name: str, **kwargs: Any) -> str:
        """将 Prompt 模板与 Variables、Examples 组合成最终输入"""
        template, examples = self.get_with_examples(name)

        # 对所有输入进行清洗
        sanitized_kwargs = {
            k: self.sanitize_input(str(v), k)
            for k, v in kwargs.items()
        }

        # 填充变量
        prompt = template.format(**sanitized_kwargs)

        # 添加 Few-shot Examples
        if examples:
            prompt += "\n\n## 示例：\n"
            for ex in examples:
                prompt += f"输入：{ex['input']}\n输出：{ex['output']}\n\n"

        # 最终长度检查
        if len(prompt) > MAX_PROMPT_LENGTH:
            prompt = prompt[:MAX_PROMPT_LENGTH]
            logger.warning(f"最终 Prompt 超过 {MAX_PROMPT_LENGTH} 字符，已截断")

        return prompt
```

**Examples 示例**：
```sql
-- 生成节点的正向示例
INSERT INTO prompt_examples (prompt_name, example_input, example_output, sort_order) VALUES
('generator', '新品发布：智能手表X', '🎉 智能手表X全新上市！\n\n⏰ 7天超长续航 | 💧 50米防水 | ❤️ 24h心率监测\n\n立即购买，享首发优惠！', 1),

('generator', '节日促销：双11狂欢', '🛒 双11狂欢节来啦！\n\n全场5折起 | 满减不封顶 | 限时秒杀\n\n错过等一年，点击立即抢购→', 2);
```

**完整的 Prompt + Examples 组合**：
```python
# 生成时调用
final_prompt = prompt_manager.format_prompt(
    "generator",
    plan="目标受众：年轻人；核心卖点：性价比",
    style="活泼",
    template="简洁文案模板"
)
# 输出：
# 你是一个营销文案专家...
# [Prompt 模板内容]
#
# ## 示例：
# 输入：新品发布：智能手表X
# 输出：🎉 智能手表X全新上市！...

### 3.14 Rate Limiting（API 限流）

防止恶意刷接口导致 API Key 被封或产生高额费用：

**限流策略**：
| 场景 | 限制 |
|------|------|
| 创建选题 | 每 IP 每分钟 5 次 |
| 用户确认 | 每 IP 每分钟 10 次 |
| 搜索知识库 | 每 IP 每分钟 20 次 |

**后端抽象**（MVP 用内存实现，预留 Redis 切换）：

```python
# api/middleware.py
from typing import Protocol

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
    """基于内存的限流实现（MVP 默认）"""

    def __init__(self):
        self._requests: dict[str, list[float]] = {}

    async def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        if key in self._requests:
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

    async def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        if key not in self._requests:
            return max_requests
        now = time.time()
        count = sum(1 for t in self._requests[key] if now - t < window_seconds)
        return max(0, max_requests - count)


# 生产环境切换 Redis（示例接口，具体实现使用 redis-py）
# class RedisRateLimiter:
#     def __init__(self, redis_url: str):
#         self._redis = redis.from_url(redis_url)
#
#     async def is_rate_limited(self, key, max_requests, window_seconds):
#         current = await self._redis.incr(key)
#         if current == 1:
#             await self._redis.expire(key, window_seconds)
#         return current > max_requests


# 全局限流器实例
rate_limiter: RateLimiterBackend = MemoryRateLimiter()


def rate_limit(max_requests: int = 5, window_seconds: int = 60):
    """装饰器：每分钟最多 N 次请求（通过 RateLimiterBackend 实现）"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            limited = await rate_limiter.is_rate_limited(
                client_ip, max_requests, window_seconds
            )
            if limited:
                remaining = await rate_limiter.get_remaining(
                    client_ip, max_requests, window_seconds
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"请求过于频繁，请 {window_seconds} 秒后重试",
                    headers={"X-RateLimit-Remaining": str(remaining)}
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# 使用
@app.post("/api/topics")
@rate_limit(max_requests=5, window_seconds=60)
async def create_topic(request: Request, topic: CreateTopicRequest):
    # ...
```

---

### 3.15 测试策略

Agent 系统的测试分为四层，从确定性逻辑到端到端验证逐层递进：

| 层级 | 测试内容 | 工具 | 运行频率 |
|------|---------|------|---------|
| **单元测试** | State 序列化/反序列化、熔断逻辑、`sanitize_input` 清洗、收敛判据、超时计算 | `pytest` | 每次 commit |
| **节点测试** | 单个节点输入输出（mock LLM 响应）、工具调用错误隔离 | `pytest` + `unittest.mock` | 每次 commit |
| **集成测试** | 完整工作流跑通（真实 LLM，限 1 次迭代）、数据库读写 | `pytest` + 真实 API Key | 每次 PR |
| **E2E** | Streamlit UI 交互流程、SSE 事件接收 | `Playwright` / 手动 | 发版前 |

**单元测试示例**：

```python
# tests/test_state.py
import pytest
from core.agent.state import AgentState, PlanState, ReflectionState

class TestAgentState:
    def test_is_iteration_exceeded(self):
        state = AgentState(max_iterations=3, current_iteration=2)
        assert not state.is_iteration_exceeded()
        state.current_iteration = 3
        assert state.is_iteration_exceeded()

    def test_is_confirmation_timed_out(self):
        from datetime import datetime, timezone, timedelta
        state = AgentState(
            waiting_since=datetime.now(timezone.utc) - timedelta(minutes=31),
            confirmation_timeout_minutes=30
        )
        assert state.is_confirmation_timed_out()

    def test_is_reflection_converged(self):
        state = AgentState(convergence_threshold=0.9)
        state.reflection_state.updated_plan = "目标受众：年轻人，核心卖点：性价比"
        state.reflection_state.previous_rejected_plan = "目标受众：年轻人，核心卖点：性价比极高"
        # 相似度 > 0.9，应判定为收敛
        assert state.is_reflection_converged()
```

**节点测试示例**：

```python
# tests/test_nodes.py
from unittest.mock import patch, MagicMock

def test_call_tool_safely_handles_network_error():
    with patch("core.agent.nodes.get_tool") as mock_get:
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = NetworkError("timeout")
        mock_get.return_value = mock_tool

        result = call_tool_safely("search_web", {"query": "test"})
        assert result["success"] is False
        assert "网络错误" in result["error"]

@patch("core.agent.nodes.llm")
def test_planner_node_sets_waiting_since(mock_llm):
    mock_llm.generate.return_value = MagicMock(
        content="策划方案内容...",
        tool_calls=None
    )
    state = AgentState(user_input="新品发布")
    result = planner_node(state)

    assert result.plan_state.plan == "策划方案内容..."
    assert result.status == "waiting_confirm"
    assert result.waiting_since is not None
```

**集成测试示例**：

```python
# tests/test_workflow.py
import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_full_workflow_single_iteration():
    """完整工作流通一次（需要真实 API Key）"""
    from core.agent.graph import create_workflow

    workflow = create_workflow()
    state = AgentState(
        user_input="新品发布：智能手表",
        max_iterations=1,  # 限 1 次迭代，控制成本
    )

    result = workflow.invoke(state)

    # 应该生成内容并走到审核阶段
    assert result.status in ("waiting_confirm", "completed", "running")
    assert result.plan_state.plan is not None
```

**测试配置**：
```bash
# pytest.ini
[pytest]
markers =
    slow: 需要真实 LLM 调用，较慢
    integration: 集成测试
```

---

## 4. API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/topics` | 创建选题，启动工作流 |
| `GET` | `/api/topics/{id}` | 获取选题及当前状态 |
| `POST` | `/api/topics/{id}/events` | **SSE 事件流**（实时推送状态） |
| `POST` | `/api/topics/{id}/confirm` | 用户确认（采纳/不采纳+意见） |
| `POST` | `/api/templates` | 创建模板 |
| `GET` | `/api/templates` | 获取模板列表 |
| `POST` | `/api/knowledge` | 添加品牌知识 |
| `GET` | `/api/knowledge/search` | 搜索品牌知识 |
| `POST` | `/api/prompts` | 创建/更新 Prompt（含 Examples） |
| `GET` | `/api/prompts/{name}` | 获取 Prompt 及关联 Examples |

---

## 5. 附录

### 5.1 防御清单（实现自查）

> 详细实现约束见 `Constraint.md`，以下为架构层面的关键防御点汇总。

| 场景 | 风险 | 防御措施 | 对应章节 |
|------|------|----------|----------|
| 用户输入超长 | 上下文溢出 | `sanitize_input()` 截断 | 3.12 |
| 提示词注入 | Prompt 被劫持 | `FORBIDDEN_PATTERNS` 过滤 | 3.12 |
| LLM 幻觉 | 生成违规内容 | 双重敏感词拦截 | 3.3 |
| 死循环 | 无限迭代烧钱 | `max_iterations` 熔断 | 3.7 |
| 无效反思循环 | 反思不收敛持续消耗 token | `is_reflection_converged()` 收敛检测 | 3.6 |
| 用户确认超时 | 资源长期占用 | `is_confirmation_timed_out()` 30 分钟超时 | 3.7 |
| API 被刷 | API Key 被封 | Rate Limiting + `RateLimiterBackend` 抽象 | 3.14 |
| 工具调用失败 | Agent 崩溃 | `call_tool_safely()` 错误隔离 | 3.8 |
| 启动时无校验 | 运行中才发现配置错误 | Pydantic `field_validator` | 3.9 |
| 迭代历史丢失 | 无法审计和复盘 | `topic_iterations` 表记录每轮快照 | 3.11 |
| SSE 隐式轮询 | CPU 空转浪费 | `asyncio.Queue` 事件总线 | 3.10 |

### 5.2 与 Constraint.md 的关联

`Constraint.md` 是本架构文档的实现指南，定义了具体的编码规范：

- **状态管理**：必须使用 Pydantic `BaseModel`，禁止 `TypedDict`
- **熔断机制**：每次迭代前检查 `is_iteration_exceeded()`，不超过 `max_iterations`
- **错误处理**：所有工具调用必须经过 `call_tool_safely()`
- **Prompt 治理**：所有外部输入必须经过 `sanitize_input()` 清洗
- **环境校验**：启动时校验关键配置，使用 `field_validator`
- **前端交互**：SSE 而非轮询

两者应配合阅读：ARCHITECTURE.md 定义"是什么"和"为什么"，Constraint.md 定义"怎么做"和"禁止做什么"。

---

> 最后提醒：**MVP 的核心是跑通工作流，不要过度设计**。
> 遇到复杂需求，先问自己："这是 MVP 必须的吗？"
