"""
SQLAlchemy 数据模型
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 基础类"""
    pass


class Topic(Base):
    """选题模型"""
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    user_input: Mapped[str] = mapped_column(Text)

    # 策划阶段
    plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plan_approved: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    plan_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 生成阶段
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_approved: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    content_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 审核阶段
    review_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    review_approved: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # 反思阶段
    reflection: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)

    # 元数据
    status: Mapped[str] = mapped_column(String(50), default="running")
    fail_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    current_step: Mapped[str] = mapped_column(String(50), default="planning")
    waiting_since: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # 进入 waiting_confirm 的时间戳

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "user_input": self.user_input,
            "plan": self.plan,
            "plan_approved": self.plan_approved,
            "plan_feedback": self.plan_feedback,
            "content": self.content,
            "content_approved": self.content_approved,
            "content_feedback": self.content_feedback,
            "review_result": self.review_result,
            "review_approved": self.review_approved,
            "reflection": self.reflection,
            "updated_plan": self.updated_plan,
            "iteration_count": self.iteration_count,
            "status": self.status,
            "fail_reason": self.fail_reason,
            "current_step": self.current_step,
            "waiting_since": self.waiting_since.isoformat() if self.waiting_since else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TopicIteration(Base):
    """
    迭代历史模型

    记录每次迭代的完整快照，支持审计和复盘。
    每次审核不通过→反思→重新生成时写入一条记录。
    """
    __tablename__ = "topic_iterations"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("topics.id"), nullable=False, index=True
    )
    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)

    # 该轮迭代的输入输出快照
    plan_snapshot: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # 当轮策划方案
    content_snapshot: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # 当轮生成内容
    review_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # 审核结果
    reflection_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # 反思分析

    # 该轮元数据
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending / approved / rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Template(Base):
    """内容模板模型"""
    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    content: Mapped[str] = mapped_column(Text)  # 含 {变量} 占位符
    style: Mapped[str] = mapped_column(String(50))  # 风格：专业/活泼/感性/幽默
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeItem(Base):
    """品牌知识模型"""
    __tablename__ = "knowledge_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(100))  # 分类：品牌/产品/行业
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Prompt(Base):
    """Prompt 模板模型"""
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    content: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PromptExample(Base):
    """Prompt 示例模型（Few-shot）"""
    __tablename__ = "prompt_examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_name: Mapped[str] = mapped_column(String(100))  # 关联的 prompt
    example_input: Mapped[str] = mapped_column(Text)
    example_output: Mapped[str] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
