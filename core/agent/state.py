"""
Pydantic State 模型 - Agent 核心状态定义

使用 pydantic.BaseModel 管理所有状态，符合 Constraint.md 要求：
- 不支持 TypedDict 或 dataclass
- 内置序列化/反序列化（.model_dump_json()）
- 自动校验类型
"""
from datetime import datetime, timezone
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict, Field


class PlanState(BaseModel):
    """策划阶段状态"""
    plan: str | None = None
    approved: bool | None = None
    feedback: str | None = None


class ContentState(BaseModel):
    """生成阶段状态"""
    content: str | None = None
    approved: bool | None = None
    feedback: str | None = None


class ReviewState(BaseModel):
    """审核阶段状态"""
    result: str | None = None          # 通过/驳回 + 理由
    approved: bool | None = None


class ReflectionState(BaseModel):
    """反思阶段状态"""
    analysis: str | None = None        # 反思分析结果
    updated_plan: str | None = None    # 更新后的策划方案
    previous_rejected_plan: str | None = None  # 上一轮被驳回的 plan，用于收敛检测
    iteration_count: int = 0


class ToolCallRecord(BaseModel):
    """工具调用记录"""
    tool_name: str
    args: dict
    result: object
    timestamp: str = ""


class AgentState(BaseModel):
    """
    主状态类 - 所有阶段的容器

    状态流转：planning → generating → reviewing → reflection → 循环或结束
    status: running → waiting_confirm → completed / failed
    """

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
    step: str = "planning"  # planning / generating / reviewing / reflection

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
    confirmation_timeout_minutes: int = 30  # 确认超时时间

    # 关联的数据库 ID（用于状态持久化）
    topic_id: int | None = None

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
        """
        检查是否超过最大迭代次数

        熔断机制：current_iteration >= max_iterations 时返回 True，
        防止无限迭代烧钱
        """
        return self.current_iteration >= self.max_iterations

    def is_confirmation_timed_out(self) -> bool:
        """
        检查用户确认是否超时

        从 waiting_since 开始计算，超过 confirmation_timeout_minutes 分钟认为超时。
        超时时应释放 SSE 连接和数据库资源。
        """
        if self.waiting_since is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.waiting_since).total_seconds()
        return elapsed > self.confirmation_timeout_minutes * 60

    def is_reflection_converged(self) -> bool:
        """
        检查反思是否已收敛

        比较当前 updated_plan 和 previous_rejected_plan 的相似度。
        相似度 >= convergence_threshold 时认为已经收敛，提前终止避免无效循环。
        """
        current = self.reflection_state.updated_plan
        previous = self.reflection_state.previous_rejected_plan
        if not current or not previous:
            return False
        similarity = SequenceMatcher(None, current, previous).ratio()
        return similarity >= self.convergence_threshold

    model_config = ConfigDict(arbitrary_types_allowed=True)