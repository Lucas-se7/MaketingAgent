"""
LangGraph 节点定义

包含：planner / generator / reviewer / reflection / user_confirm
所有工具调用必须经过 call_tool_safely()（Constaint.md 要求）
"""
import logging
from datetime import datetime, timezone

from core.agent.state import AgentState, ToolCallRecord
from core.agent.prompts import prompt_manager
from core.llm import get_llm_client, get_llm_model, generate_with_tracing
from core.tools import check_sensitive, check_sensitive_with_filter

logger = logging.getLogger(__name__)


# ============================================================================
# 异常类（Constaint.md 熔断机制）
# ============================================================================


class MaxIterationsExceededError(Exception):
    """
    超过最大迭代次数异常

    当 current_iteration >= max_iterations 时抛出，
    防止无限迭代烧钱
    """
    pass


class ReflectionConvergedError(Exception):
    """
    反思收敛异常

    连续两次 updated_plan 相似度 >= convergence_threshold 时抛出，
    表示反思已无改进空间，提前终止
    """
    pass


class WorkflowInterruptError(Exception):
    """
    工作流中断异常

    用于在需要用户确认时暂停工作流，
    由 API 层捕获并处理
    """
    pass


# ============================================================================
# 工具调用安全包装
# ============================================================================


def call_tool_safely(tool_name: str, args: dict) -> dict:
    """
    安全调用工具，失败时返回错误信息而非崩溃

    Constaint.md 要求：所有工具调用必须经过此函数统一捕获异常。

    Args:
        tool_name: 工具名称
        args: 工具参数

    Returns:
        {"success": bool, "result": any, "error": str}
    """
    try:
        # MVP 阶段只有敏感词检测工具
        if tool_name == "check_sensitive":
            result = check_sensitive(args.get("content", ""))
            return {"success": True, "result": result.to_dict()}

        # 预留：其他工具扩展
        raise NotImplementedError(f"Tool '{tool_name}' not implemented")

    except NotImplementedError:
        # 预期的未实现工具，直接抛出
        raise

    except Exception as e:
        logger.error(f"工具调用失败: {tool_name}, error={e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# 工作流节点
# ============================================================================


def planner_node(state: AgentState) -> AgentState:
    """
    策划节点

    根据用户选题生成策划方案。
    Agent 可主动调用知识库搜索工具。

    Args:
        state: 当前 AgentState

    Returns:
        更新后的 AgentState（plan_state.plan 已填充）
    """
    logger.info("PlannerNode: 生成策划方案")

    client = get_llm_client()
    model = get_llm_model()

    # 构造 Prompt（经过 sanitize_input 清洗）
    prompt = prompt_manager.format_prompt(
        "planner",
        user_input=state.user_input
    )

    # 如果是驳回后重新策划，附加上一轮的反馈和方案
    if state.plan_state.feedback:
        old_plan = state.plan_state.plan or ""
        prompt += (
            f"\n\n## 用户驳回了上一版策划方案，请重新制定\n"
            f"上一版方案：{old_plan}\n"
            f"用户反馈：{state.plan_state.feedback}\n"
            f"请根据反馈意见重新制定策划方案，务必与上一版有明显区别。"
        )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    plan = generate_with_tracing(client, messages, model=model)

    # 更新状态
    state.plan_state.plan = plan
    state.step = "planning"

    logger.info(f"PlannerNode: 策划方案生成完成，长度={len(plan)}")
    return state


def generator_node(state: AgentState) -> AgentState:
    """
    生成节点

    根据策划方案生成营销文案。
    使用双重敏感词拦截的第一道：生成后立即过滤。

    Args:
        state: 当前 AgentState

    Returns:
        更新后的 AgentState（content_state.content 已填充）
    """
    logger.info("GeneratorNode: 生成营销文案")

    client = get_llm_client()
    model = get_llm_model()

    # 使用更新后的策划方案（如果有）
    plan = state.reflection_state.updated_plan or state.plan_state.plan

    # 获取用户反馈（如果是重新生成）
    feedback = state.content_state.feedback or ""

    # 构造 Prompt
    prompt = prompt_manager.format_prompt(
        "generator",
        plan=plan or "",
        style="活泼",  # MVP 固定风格
        template="简洁文案模板"
    )

    # 如果是驳回后重新生成，附加上一版内容和反馈
    if feedback:
        old_content = state.content_state.content or ""
        prompt += (
            f"\n\n## 用户驳回了上一版内容，请重新生成\n"
            f"被驳回的内容：{old_content}\n"
            f"用户反馈：{feedback}\n"
            f"请根据反馈意见重新生成，务必与上一版有明显不同。"
        )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    content = generate_with_tracing(client, messages, model=model)

    # 双重敏感词拦截 - 第一道：Generator 输出后立即过滤
    filtered_content, check_result = check_sensitive_with_filter(content)
    if not check_result.is_pass:
        logger.warning(
            f"GeneratorNode: 检测到敏感词，已过滤 - {check_result.detected_words}"
        )

    state.content_state.content = filtered_content
    state.step = "generating"

    logger.info(f"GeneratorNode: 文案生成完成，长度={len(filtered_content)}")
    return state


def reviewer_node(state: AgentState) -> AgentState:
    """
    审核节点

    审核生成的文案是否符合品牌调性、是否有敏感内容。
    使用双重敏感词拦截的第二道：Reviewer 审核时二次检测。

    Args:
        state: 当前 AgentState

    Returns:
        更新后的 AgentState（review_state.result + review_state.approved 已填充）
    """
    logger.info("ReviewerNode: 审核文案")

    client = get_llm_client()
    model = get_llm_model()

    # 构造 Prompt
    prompt = prompt_manager.format_prompt(
        "reviewer",
        content=state.content_state.content or "",
        knowledge="品牌调性：年轻、活力、创新"  # MVP 简化
    )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    review_result = generate_with_tracing(client, messages, model=model)

    # 双重敏感词拦截 - 第二道：Reviewer 审核时检测
    check_result = check_sensitive(state.content_state.content or "")
    if not check_result.is_pass:
        review_result = f"【敏感词检测未通过】{check_result.message}\n\n{review_result}"
        logger.warning(
            f"ReviewerNode: 敏感词检测未通过 - {check_result.detected_words}"
        )

    # 判断是否通过（根据 LLM 输出的关键词）
    is_approved = "通过" in review_result and "驳回" not in review_result

    state.review_state.result = review_result
    state.review_state.approved = is_approved
    state.step = "reviewing"

    logger.info(f"ReviewerNode: 审核完成，通过={is_approved}")
    return state


def reflection_node(state: AgentState) -> AgentState:
    """
    反思节点

    分析驳回原因，更新策划方案。

    包含熔断机制（Constaint.md 要求）：
    1. 迭代次数检查：current_iteration >= max_iterations → 抛出 MaxIterationsExceededError
    2. 收敛检测：连续两次 updated_plan 相似度 >= convergence_threshold →
       抛出 ReflectionConvergedError

    Args:
        state: 当前 AgentState

    Returns:
        更新后的 AgentState

    Raises:
        MaxIterationsExceededError: 超过最大迭代次数
        ReflectionConvergedError: 反思已收敛，无需继续
    """
    logger.info("ReflectionNode: 反思并更新策划方案")

    # --- 熔断检查 1：迭代次数 ---
    if state.is_iteration_exceeded():
        state.status = "failed"
        state.fail_reason = (
            f"超过最大迭代次数 ({state.max_iterations})，"
            f"请人工介入调整方向"
        )
        logger.warning(state.fail_reason)
        raise MaxIterationsExceededError(state.fail_reason)

    client = get_llm_client()
    model = get_llm_model()

    # 构造 Prompt
    prompt = prompt_manager.format_prompt(
        "reflection",
        review_result=state.review_state.result or "",
        feedback=state.content_state.feedback or "",
        original_plan=state.plan_state.plan or ""
    )

    # 调用 LLM
    messages = [{"role": "user", "content": prompt}]
    reflection_result = generate_with_tracing(client, messages, model=model)

    # 保存反思分析
    state.reflection_state.analysis = reflection_result

    # 解析更新后的策划方案
    new_plan = None
    if "更新后的策划方案" in reflection_result:
        lines = reflection_result.split("\n")
        for i, line in enumerate(lines):
            if "更新后的策划方案" in line and i + 1 < len(lines):
                new_plan = "\n".join(lines[i + 1:]).strip()
                break

    if not new_plan:
        # 如果解析不到更新方案，使用原方案并附加反思
        new_plan = (state.plan_state.plan or "") + "\n\n[反思调整]\n" + reflection_result

    # --- 收敛检测：保存上一轮被驳回的 plan ---
    state.reflection_state.previous_rejected_plan = (
        state.reflection_state.updated_plan or state.plan_state.plan or ""
    )

    # 更新状态
    state.reflection_state.updated_plan = new_plan
    state.reflection_state.iteration_count += 1
    state.current_iteration += 1

    # --- 熔断检查 2：收敛检测 ---
    if state.is_reflection_converged():
        state.status = "failed"
        state.fail_reason = (
            f"反思已收敛，连续两次更新的策划方案高度相似"
            f"（相似度 ≥ {state.convergence_threshold}），"
            f"请人工介入调整方向"
        )
        logger.warning(state.fail_reason)
        raise ReflectionConvergedError(state.fail_reason)

    logger.info(
        f"ReflectionNode: 反思完成，"
        f"迭代次数={state.current_iteration}/{state.max_iterations}"
    )
    return state


def user_confirm_node(state: AgentState) -> AgentState:
    """
    用户确认节点

    暂停工作流，等待用户输入（采纳/不采纳+意见）。

    注意：此节点设置 waiting_since 时间戳，
    API 层负责检测超时（Constaint.md: 30 分钟超时）。
    """
    state.status = "waiting_confirm"
    state.waiting_since = datetime.now(timezone.utc)
    logger.info(
        f"UserConfirmNode: 等待用户确认，"
        f"当前步骤={state.step}, waiting_since={state.waiting_since}"
    )
    return state