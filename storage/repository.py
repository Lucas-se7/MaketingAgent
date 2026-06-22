"""
数据仓库层 - 状态序列化/反序列化

负责在 Pydantic AgentState 和 SQLAlchemy 数据库模型之间转换。
所有数据库操作均使用参数化查询，避免 SQL 注入。
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

from storage.models import Topic, TopicIteration
from storage.database import get_db_session
from core.agent.state import (
    AgentState,
    PlanState,
    ContentState,
    ReviewState,
    ReflectionState,
)

logger = logging.getLogger(__name__)


class TopicRepository:
    """
    Topic 数据仓库

    职责：
    - 将 AgentState Pydantic 对象序列化到数据库
    - 从数据库反序列化为 AgentState
    - 管理迭代历史快照
    """

    # --- 状态持久化 ---

    def save(self, state: AgentState) -> None:
        """
        将 AgentState 逐字段持久化到数据库（参数化查询，避免 SQL 注入）

        Args:
            state: 当前 AgentState 实例
        """
        if state.topic_id is None:
            raise ValueError("AgentState.topic_id 不能为空，无法持久化")

        with get_db_session() as session:
            topic = session.query(Topic).filter(Topic.id == state.topic_id).first()
            if not topic:
                raise ValueError(f"Topic id={state.topic_id} 不存在")

            # 序列化分状态字段
            topic.plan = state.plan_state.plan
            topic.plan_approved = state.plan_state.approved
            topic.plan_feedback = state.plan_state.feedback

            topic.content = state.content_state.content
            topic.content_approved = state.content_state.approved
            topic.content_feedback = state.content_state.feedback

            topic.review_result = state.review_state.result
            topic.review_approved = state.review_state.approved

            topic.reflection = state.reflection_state.analysis
            topic.updated_plan = state.reflection_state.updated_plan
            topic.iteration_count = state.reflection_state.iteration_count

            # 元数据
            topic.status = state.status
            topic.fail_reason = state.fail_reason
            topic.current_step = state.step
            topic.waiting_since = state.waiting_since
            topic.updated_at = datetime.now(timezone.utc)

            session.commit()
            logger.info(f"TopicRepository.save: topic_id={state.topic_id}, "
                        f"step={state.step}, status={state.status}")

    def load(self, topic_id: int) -> Optional[AgentState]:
        """
        从数据库记录反序列化为 AgentState

        Args:
            topic_id: 选题 ID

        Returns:
            AgentState 实例，如果不存在则返回 None
        """
        with get_db_session() as session:
            row = session.query(Topic).filter(Topic.id == topic_id).first()
            if not row:
                return None

            state = AgentState(
                user_input=row.user_input or "",
                topic_id=row.id,
                step=row.current_step or "planning",
                max_iterations=3,  # 从配置加载
                current_iteration=row.iteration_count or 0,
                status=row.status or "running",
                fail_reason=row.fail_reason,
                waiting_since=row.waiting_since,
            )

            # 恢复分状态
            state.plan_state = PlanState(
                plan=row.plan,
                approved=row.plan_approved,
                feedback=row.plan_feedback,
            )
            state.content_state = ContentState(
                content=row.content,
                approved=row.content_approved,
                feedback=row.content_feedback,
            )
            state.review_state = ReviewState(
                result=row.review_result,
                approved=row.review_approved,
            )
            state.reflection_state = ReflectionState(
                analysis=row.reflection,
                updated_plan=row.updated_plan,
                iteration_count=row.iteration_count or 0,
            )

            return state

    # --- 迭代历史 ---

    def save_iteration_snapshot(self, state: AgentState) -> None:
        """
        保存当前迭代的完整快照到 topic_iterations 表

        用于审计和复盘，记录每次反思→重新生成的完整过程。

        Args:
            state: 当前 AgentState 实例
        """
        if state.topic_id is None:
            raise ValueError("AgentState.topic_id 不能为空")

        with get_db_session() as session:
            snapshot = TopicIteration(
                topic_id=state.topic_id,
                iteration_num=state.current_iteration,
                plan_snapshot=state.plan_state.plan,
                content_snapshot=state.content_state.content,
                review_result=state.review_state.result,
                reflection_analysis=state.reflection_state.analysis,
                status="approved" if state.review_state.approved else "rejected",
            )
            session.add(snapshot)
            session.commit()
            logger.info(
                f"TopicRepository.save_iteration_snapshot: "
                f"topic_id={state.topic_id}, iteration={state.current_iteration}"
            )

    def get_iteration_history(self, topic_id: int) -> list[dict]:
        """
        获取指定选题的所有迭代历史（按迭代编号排序）

        Args:
            topic_id: 选题 ID

        Returns:
            迭代历史记录列表
        """
        with get_db_session() as session:
            rows = (
                session.query(TopicIteration)
                .filter(TopicIteration.topic_id == topic_id)
                .order_by(TopicIteration.iteration_num)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "topic_id": r.topic_id,
                    "iteration_num": r.iteration_num,
                    "plan_snapshot": r.plan_snapshot,
                    "content_snapshot": r.content_snapshot,
                    "review_result": r.review_result,
                    "reflection_analysis": r.reflection_analysis,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]


# 全局仓库实例
topic_repo = TopicRepository()