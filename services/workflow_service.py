"""
工作流执行服务

负责：
- 异步执行工作流步骤
- 事件发布到 SSE 客户端
- 状态持久化
- 熔断和超时处理

这是 API 层和 Agent 层之间的业务逻辑层。
"""
import asyncio
import logging
from datetime import datetime, timezone

from core.agent.state import AgentState
from core.agent.nodes import (
    planner_node,
    generator_node,
    reviewer_node,
    reflection_node,
    MaxIterationsExceededError,
    ReflectionConvergedError,
)
from storage import db, topic_repo
from config import settings

logger = logging.getLogger(__name__)


class WorkflowService:
    """
    工作流执行服务

    管理从选题创建到内容完成（或失败）的完整生命周期。

    用法:
        service = WorkflowService(publish_event_callback)
        await service.start_workflow(topic_id, user_input)
        await service.continue_workflow(topic_id, approved, feedback)
    """

    def __init__(self, publish_event_callback=None):
        """
        Args:
            publish_event_callback: async callable(topic_id, event_dict)
                                    用于推送 SSE 事件到客户端
        """
        self._publish = publish_event_callback or (lambda tid, ev: None)
        self._states: dict[int, AgentState] = {}

    async def start_workflow(self, topic_id: int, user_input: str) -> AgentState:
        """
        启动新的工作流

        Args:
            topic_id: 数据库中的选题 ID
            user_input: 用户输入的选题描述

        Returns:
            初始化的 AgentState
        """
        # 创建初始状态
        state = AgentState(
            user_input=user_input,
            topic_id=topic_id,
            max_iterations=settings.MAX_ITERATIONS,
            status="running",
            step="planning",
        )
        self._states[topic_id] = state

        logger.info(f"WorkflowService: 启动工作流 topic_id={topic_id}")

        # 执行第一步：策划
        await self._execute_step(state)

        return state

    async def continue_workflow(
        self,
        topic_id: int,
        approved: bool,
        feedback: str | None = None,
    ) -> AgentState:
        """
        用户确认后继续工作流

        Args:
            topic_id: 选题 ID
            approved: 是否采纳
            feedback: 驳回时的反馈意见

        Returns:
            更新后的 AgentState
        """
        state = self._states.get(topic_id)
        if not state:
            # 从数据库恢复
            state = topic_repo.load(topic_id)
            if not state:
                raise ValueError(f"Topic {topic_id} not found")
            self._states[topic_id] = state

        # 根据当前步骤应用用户确认
        if state.step == "planning":
            state.plan_state.approved = approved
            state.plan_state.feedback = feedback
            if approved:
                state.step = "generating"

        elif state.step == "generating":
            state.content_state.approved = approved
            state.content_state.feedback = feedback
            if approved:
                state.step = "reviewing"

        elif state.step == "reviewing":
            state.review_state.approved = approved
            if not approved and feedback:
                state.content_state.feedback = feedback

        # 恢复运行状态
        state.status = "running"
        state.waiting_since = None

        # 持久化
        topic_repo.save(state)

        # 推送确认事件
        await self._publish(topic_id, {
            "status": "running",
            "step": state.step,
            "result": state.get_current_result(),
        })

        # 继续执行后续步骤
        await self._execute_step(state)

        return state

    async def _execute_step(self, state: AgentState) -> None:
        """
        执行工作流步骤的主循环

        根据当前 step 依次执行节点，在需要用户确认时暂停。
        """
        topic_id = state.topic_id
        if topic_id is None:
            return

        try:
            while state.status == "running":
                # 检查超时
                if state.is_confirmation_timed_out():
                    state.status = "failed"
                    state.fail_reason = (
                        f"用户确认超时（{state.confirmation_timeout_minutes} 分钟）"
                    )
                    logger.warning(f"WorkflowService: {state.fail_reason}")
                    break

                if state.step == "planning":
                    await self._run_planner(state)

                elif state.step == "generating":
                    await self._run_generator(state)

                elif state.step == "reviewing":
                    done = await self._run_reviewer(state)
                    if done:
                        continue  # 需要反思或已完成
                    break  # 等待用户确认

                elif state.step == "reflection":
                    await self._run_reflection(state)
                    # 反思后自动重新生成
                    state.step = "generating"
                    continue

                else:
                    logger.warning(f"未知步骤: {state.step}")
                    break

        except MaxIterationsExceededError as e:
            state.status = "failed"
            state.fail_reason = str(e)
            logger.warning(f"WorkflowService: 熔断 - {e}")

        except ReflectionConvergedError as e:
            state.status = "failed"
            state.fail_reason = str(e)
            logger.warning(f"WorkflowService: 收敛 - {e}")

        except Exception as e:
            state.status = "failed"
            state.fail_reason = str(e)
            logger.error(f"WorkflowService: 工作流异常 - {e}", exc_info=True)

        finally:
            # 持久化最新状态
            topic_repo.save(state)

            # 仅对终态（completed / failed）推送事件；
            # running / waiting_confirm 由各 step 方法自行推送，
            # 避免重复事件中携带的 fail_reason: None 混淆日志
            if state.status in ("completed", "failed"):
                await self._publish(topic_id, {
                    "status": state.status,
                    "step": state.step,
                    "result": state.get_current_result(),
                    "fail_reason": state.fail_reason,
                    "content": state.content_state.content,
                })
                self._states.pop(topic_id, None)

    async def _run_planner(self, state: AgentState) -> None:
        """执行策划节点"""
        logger.info(f"WorkflowService: 执行策划 topic_id={state.topic_id}")

        await self._publish(state.topic_id, {
            "status": "running",
            "step": "planning",
            "result": "[策划中...]",
        })

        state = planner_node(state)
        topic_repo.save(state)

        # 进入等待确认
        state.status = "waiting_confirm"
        state.waiting_since = datetime.now(timezone.utc)
        topic_repo.save(state)

        await self._publish(state.topic_id, {
            "status": "waiting_confirm",
            "step": "planning",
            "result": state.plan_state.plan or "",
        })

    async def _run_generator(self, state: AgentState) -> None:
        """执行生成节点"""
        logger.info(f"WorkflowService: 执行生成 topic_id={state.topic_id}")

        await self._publish(state.topic_id, {
            "status": "running",
            "step": "generating",
            "result": "[生成中...]",
        })

        state = generator_node(state)
        topic_repo.save(state)

        # 进入等待确认
        state.status = "waiting_confirm"
        state.waiting_since = datetime.now(timezone.utc)
        topic_repo.save(state)

        await self._publish(state.topic_id, {
            "status": "waiting_confirm",
            "step": "generating",
            "result": state.content_state.content or "",
        })

    async def _run_reviewer(self, state: AgentState) -> bool:
        """
        执行审核节点

        Returns:
            True: 工作流需要继续（不通过→反思，或已通过→完成）
            False: 等待用户确认
        """
        logger.info(f"WorkflowService: 执行审核 topic_id={state.topic_id}")

        await self._publish(state.topic_id, {
            "status": "running",
            "step": "reviewing",
            "result": "[审核中...]",
        })

        state = reviewer_node(state)

        # 保存迭代快照
        topic_repo.save_iteration_snapshot(state)
        topic_repo.save(state)

        if state.review_state.approved:
            state.status = "completed"
            topic_repo.save(state)
            return True

        # 审核不通过：检查是否可以进入反思
        if state.is_iteration_exceeded():
            state.status = "failed"
            state.fail_reason = (
                f"超过最大迭代次数 ({state.max_iterations})"
            )
            topic_repo.save(state)
            return True

        # 进入等待确认（让用户看到审核结果）
        state.status = "waiting_confirm"
        state.waiting_since = datetime.now(timezone.utc)
        topic_repo.save(state)

        await self._publish(state.topic_id, {
            "status": "waiting_confirm",
            "step": "reviewing",
            "result": state.review_state.result or "",
        })

        return False

    async def _run_reflection(self, state: AgentState) -> None:
        """执行反思节点"""
        logger.info(f"WorkflowService: 执行反思 topic_id={state.topic_id}")

        await self._publish(state.topic_id, {
            "status": "running",
            "step": "reflection",
            "result": "[反思分析中...]",
        })

        state = reflection_node(state)
        topic_repo.save(state)

        await self._publish(state.topic_id, {
            "status": "running",
            "step": state.step,
            "result": state.reflection_state.analysis or "",
        })

    def get_state(self, topic_id: int) -> AgentState | None:
        """获取内存中的工作流状态"""
        return self._states.get(topic_id)


# 全局工作流服务实例（启动时通过 create_routes 注入 publish_event）
workflow_service = WorkflowService()