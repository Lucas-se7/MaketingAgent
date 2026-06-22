"""
LangGraph 工作流编排

策划 → 用户确认 → 生成 → 用户确认 → 审核 → 用户确认 → END
                              ↑                          ↓
                              └── 反思 ← 审核不通过 ←───┘

包含熔断机制（Constaint.md 要求）：
- 迭代次数超限检查
- 收敛检测
- 用户确认超时处理
"""
import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.agent.state import AgentState
from core.agent.nodes import (
    planner_node,
    generator_node,
    reviewer_node,
    reflection_node,
    user_confirm_node,
    MaxIterationsExceededError,
    ReflectionConvergedError,
    WorkflowInterruptError,
)

logger = logging.getLogger(__name__)


def create_workflow(checkpointer=None):
    """
    创建工作流图

    Flow:
        START → [planner] → [user_confirm] → [generator] → [user_confirm]
                              ↓ (采纳)
                        [reviewer] → [user_confirm]
                              ↓ (不通过)        ↓ (通过)
                        [reflection] → END     END

    Args:
        checkpointer: 状态持久化检查点（MVP 使用内存）

    Returns:
        编译后的 LangGraph 工作流
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("user_confirm", user_confirm_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("reflection", reflection_node)

    # 设置起点
    workflow.set_entry_point("planner")

    # --- 边定义 ---
    # planner → user_confirm
    workflow.add_edge("planner", "user_confirm")

    # generator → user_confirm
    workflow.add_edge("generator", "user_confirm")

    # reviewer → user_confirm
    workflow.add_edge("reviewer", "user_confirm")

    # reflection → generator（反思后重新生成）
    workflow.add_edge("reflection", "generator")

    # --- 条件边：用户确认后决定下一步 ---
    def should_continue(
        state: AgentState,
    ) -> Literal["reflection", "generator", "user_confirm", END]:
        """
        根据审核结果和用户确认决定下一步

        熔断逻辑（Constaint.md 要求）：
        1. 用户确认超时 → END（标记 failed）
        2. 审核通过 → END（标记 completed）
        3. 迭代超限 → END（标记 failed）
        4. 反思收敛 → END（标记 failed）
        5. 用户要求重新生成 → generator
        6. 审核不通过 → reflection
        """
        # 检查确认超时
        if state.is_confirmation_timed_out():
            state.status = "failed"
            state.fail_reason = (
                f"用户确认超时（{state.confirmation_timeout_minutes} 分钟），"
                f"已释放资源"
            )
            logger.warning(state.fail_reason)
            return END

        # 如果状态是 waiting_confirm，继续等待
        if state.status == "waiting_confirm":
            return "user_confirm"

        # 审核结果判断
        if state.review_state.approved:
            state.status = "completed"
            logger.info("审核通过，工作流完成")
            return END

        # 审核不通过
        if state.step != "reviewing":
            # 非审核步骤的确认，继续到下一步
            return "user_confirm"

        # 检查迭代次数
        if state.is_iteration_exceeded():
            state.status = "failed"
            state.fail_reason = (
                f"超过最大迭代次数 ({state.max_iterations})，"
                f"请人工介入"
            )
            logger.warning(state.fail_reason)
            return END

        # 检查收敛
        if state.is_reflection_converged():
            state.status = "failed"
            state.fail_reason = (
                f"反思已收敛（相似度 ≥ {state.convergence_threshold}），"
                f"请人工介入调整方向"
            )
            logger.warning(state.fail_reason)
            return END

        # 用户要求直接重新生成
        if state.content_state.feedback and "重新生成" in state.content_state.feedback:
            logger.info("用户要求直接重新生成")
            return "generator"

        # 进入反思
        logger.info("审核不通过，进入反思闭环")
        return "reflection"

    workflow.add_conditional_edges(
        "user_confirm",
        should_continue,
        {
            "reflection": "reflection",
            "generator": "generator",
            "user_confirm": "user_confirm",
            END: END,
        },
    )

    # 编译
    checkpointer = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=checkpointer)