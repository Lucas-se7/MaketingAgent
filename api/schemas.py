"""
API Pydantic Schemas

定义所有请求/响应的数据结构，与 Constraint.md 保持一致。
使用 Pydantic 的 field_validator 进行输入校验。
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- 请求模型 ---


class CreateTopicRequest(BaseModel):
    """创建选题请求"""
    user_input: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户输入的选题描述"
    )

    @field_validator("user_input")
    @classmethod
    def sanitize_user_input(cls, v: str) -> str:
        """对用户输入进行基础清洗（详细清洗在 sanitize_input 中）"""
        # 去除首尾空白
        v = v.strip()
        # 去除多余空白
        import re
        v = re.sub(r"\s+", " ", v)
        return v


class ConfirmRequest(BaseModel):
    """用户确认请求"""
    approved: bool = Field(..., description="是否采纳")
    feedback: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="驳回时的反馈意见"
    )

    @field_validator("feedback")
    @classmethod
    def sanitize_feedback(cls, v: Optional[str]) -> Optional[str]:
        """对用户反馈进行基础清洗"""
        if v is None:
            return v
        v = v.strip()
        if len(v) > 2000:
            v = v[:2000]
        return v


class CreateTemplateRequest(BaseModel):
    """创建模板请求"""
    name: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=5000)
    style: str = Field(..., min_length=1, max_length=50)


class CreateKnowledgeRequest(BaseModel):
    """添加品牌知识请求"""
    category: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=10000)


class CreatePromptRequest(BaseModel):
    """创建/更新 Prompt 请求"""
    name: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=20000)


# --- 响应模型 ---


class TopicResponse(BaseModel):
    """选题状态响应"""
    id: int
    title: str
    user_input: str

    # 策划阶段
    plan: Optional[str] = None
    plan_approved: Optional[bool] = None
    plan_feedback: Optional[str] = None

    # 生成阶段
    content: Optional[str] = None
    content_approved: Optional[bool] = None
    content_feedback: Optional[str] = None

    # 审核阶段
    review_result: Optional[str] = None
    review_approved: Optional[bool] = None

    # 反思阶段
    reflection: Optional[str] = None
    updated_plan: Optional[str] = None
    iteration_count: int = 0

    # 元数据
    status: str = "running"
    fail_reason: Optional[str] = None
    current_step: str = "planning"
    waiting_since: Optional[datetime] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CreateTopicResponse(BaseModel):
    """创建选题响应"""
    topic_id: int
    status: str


class ConfirmResponse(BaseModel):
    """确认响应"""
    status: str


class TemplateResponse(BaseModel):
    """模板响应"""
    id: int
    name: str
    content: str
    style: str
    created_at: Optional[str] = None


class KnowledgeResponse(BaseModel):
    """知识库条目响应"""
    id: int
    category: str
    content: str
    created_at: Optional[str] = None


class PromptResponse(BaseModel):
    """Prompt 响应"""
    id: int
    name: str
    content: str
    version: int
    updated_at: Optional[str] = None


class PromptDetailResponse(BaseModel):
    """Prompt 详情响应（含 Examples）"""
    id: int
    name: str
    content: str
    version: int
    examples: list[dict] = []
    updated_at: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None


class IterationHistoryResponse(BaseModel):
    """迭代历史响应"""
    topic_id: int
    iterations: list[dict]