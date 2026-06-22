"""
单元测试：节点 I/O（mock LLM）和工具调用错误隔离

Constaint.md 要求：
| 节点测试 | 单节点 I/O（mock LLM）、工具调用错误隔离 | pytest + mock | 每次 commit |
"""
import pytest
from unittest.mock import patch, MagicMock

from core.agent.state import AgentState, PlanState, ContentState, ReviewState, ReflectionState
from core.agent.nodes import (
    call_tool_safely,
    planner_node,
    generator_node,
    reviewer_node,
    reflection_node,
    user_confirm_node,
    MaxIterationsExceededError,
    ReflectionConvergedError,
    WorkflowInterruptError,
)


class TestCallToolSafely:

    def test_check_sensitive_tool(self):
        """测试敏感词工具调用"""
        result = call_tool_safely("check_sensitive", {"content": "正常内容"})
        assert result["success"] is True
        assert "result" in result
        assert result["result"]["is_pass"] is True

    def test_unknown_tool_raises(self):
        """未实现的工具抛出 NotImplementedError"""
        with pytest.raises(NotImplementedError):
            call_tool_safely("unknown_tool", {})

    def test_tool_call_error_handled(self):
        """工具调用异常被捕获"""
        result = call_tool_safely("check_sensitive", {})  # 缺少 content 参数
        # 应该不会崩溃，返回错误信息
        assert "success" in result


class TestPlannerNode:

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_planner_sets_plan_and_step(self, mock_model, mock_client, mock_generate):
        """策划节点应设置 plan 并更新 step"""
        mock_generate.return_value = "测试策划方案内容"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState(user_input="测试选题")
        result = planner_node(state)

        assert result.plan_state.plan == "测试策划方案内容"
        assert result.step == "planning"

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_planner_user_input_preserved(self, mock_model, mock_client, mock_generate):
        """策划节点不应修改 user_input"""
        mock_generate.return_value = "策划方案"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState(user_input="新品发布：智能手表")
        result = planner_node(state)

        assert result.user_input == "新品发布：智能手表"


class TestGeneratorNode:

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_generator_sets_content(self, mock_model, mock_client, mock_generate):
        """生成节点应设置 content"""
        mock_generate.return_value = "这是一条精彩的营销文案"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState(user_input="测试")
        state.plan_state.plan = "策划方案"
        result = generator_node(state)

        assert result.content_state.content is not None
        assert result.step == "generating"

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_generator_uses_updated_plan(self, mock_model, mock_client, mock_generate):
        """有 updated_plan 时应使用它"""
        mock_generate.return_value = "基于更新方案的文案"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState()
        state.plan_state.plan = "原始方案"
        state.reflection_state.updated_plan = "更新后的方案"
        result = generator_node(state)

        # 验证生成了内容
        assert result.content_state.content == "基于更新方案的文案"


class TestReviewerNode:

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_reviewer_approved(self, mock_model, mock_client, mock_generate):
        """审核通过"""
        mock_generate.return_value = "审核结果：通过。内容符合要求。"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState()
        state.content_state.content = "测试内容"
        result = reviewer_node(state)

        assert result.review_state.approved is True
        assert result.step == "reviewing"

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_reviewer_rejected(self, mock_model, mock_client, mock_generate):
        """审核驳回"""
        mock_generate.return_value = "审核结果：驳回。原因：内容不够吸引人。"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState()
        state.content_state.content = "无聊的内容"
        result = reviewer_node(state)

        assert result.review_state.approved is False
        assert result.review_state.result is not None


class TestReflectionNode:

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_reflection_updates_iteration(self, mock_model, mock_client, mock_generate):
        """反思节点应增加迭代计数"""
        mock_generate.return_value = "问题分析\n调整建议\n更新后的策划方案\n新方案内容"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState(max_iterations=3, current_iteration=0)
        state.review_state.result = "驳回：内容不符"
        state.plan_state.plan = "原方案"
        result = reflection_node(state)

        assert result.current_iteration == 1
        assert result.reflection_state.iteration_count == 1
        assert result.reflection_state.analysis is not None

    def test_reflection_max_iterations_exceeded(self):
        """超过最大迭代次数时抛出异常"""
        state = AgentState(max_iterations=3, current_iteration=3)
        state.review_state.result = "驳回"

        with pytest.raises(MaxIterationsExceededError):
            reflection_node(state)

        assert state.status == "failed"
        assert "超过最大迭代次数" in (state.fail_reason or "")

    @patch("core.agent.nodes.generate_with_tracing")
    @patch("core.agent.nodes.get_llm_client")
    @patch("core.agent.nodes.get_llm_model")
    def test_reflection_convergence_detected(self, mock_model, mock_client, mock_generate):
        """收敛检测：相似度 >= 阈值时抛出异常"""
        mock_generate.return_value = "更新后的策划方案\n新方案"
        mock_model.return_value = "test-model"
        mock_client.return_value = MagicMock()

        state = AgentState(
            max_iterations=3,
            current_iteration=0,
            convergence_threshold=0.5,  # 设置较低阈值方便测试
        )
        state.review_state.result = "驳回"
        state.plan_state.plan = "目标受众：年轻人，核心卖点：性价比"
        # 设置 updated_plan 与即将生成的方案高度相似
        state.reflection_state.updated_plan = "新方案"
        # 当 reflection_node 运行后，previous = "新方案"，updated = "新方案"
        # 相似度 = 1.0 >= 0.5，应收敛

        with pytest.raises(ReflectionConvergedError):
            reflection_node(state)

        assert state.status == "failed"
        assert "收敛" in (state.fail_reason or "")


class TestUserConfirmNode:

    def test_user_confirm_sets_status(self):
        """用户确认节点应设置 waiting_confirm 和时间戳"""
        state = AgentState(step="planning")
        result = user_confirm_node(state)

        assert result.status == "waiting_confirm"
        assert result.waiting_since is not None

    def test_user_confirm_logs_step(self):
        """确认节点应记录当前步骤"""
        state = AgentState(step="generating")
        result = user_confirm_node(state)

        assert result.status == "waiting_confirm"
        assert result.step == "generating"  # step 不应改变


class TestErrorClasses:

    def test_max_iterations_error(self):
        """MaxIterationsExceededError 可正确创建"""
        error = MaxIterationsExceededError("超过3次迭代")
        assert str(error) == "超过3次迭代"
        assert isinstance(error, Exception)

    def test_reflection_converged_error(self):
        """ReflectionConvergedError 可正确创建"""
        error = ReflectionConvergedError("反思已收敛")
        assert str(error) == "反思已收敛"
        assert isinstance(error, Exception)

    def test_workflow_interrupt_error(self):
        """WorkflowInterruptError 可正确创建"""
        error = WorkflowInterruptError()
        assert isinstance(error, Exception)