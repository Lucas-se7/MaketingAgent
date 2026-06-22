"""
单元测试：AgentState 序列化、熔断逻辑、收敛判据、超时计算

Constaint.md 要求：
| 层级 | 重点 | 工具 | 运行频率 |
|------|------|------|---------|
| 单元测试 | State 序列化、熔断逻辑、sanitize_input、收敛判据 | pytest | 每次 commit |
"""
import pytest
from datetime import datetime, timezone, timedelta

from core.agent.state import (
    AgentState,
    PlanState,
    ContentState,
    ReviewState,
    ReflectionState,
    ToolCallRecord,
)


class TestAgentState:

    def test_default_values(self):
        """测试默认值"""
        state = AgentState()
        assert state.user_input == ""
        assert state.step == "planning"
        assert state.status == "running"
        assert state.max_iterations == 3
        assert state.current_iteration == 0
        assert state.convergence_threshold == 0.9
        assert state.confirmation_timeout_minutes == 30
        assert state.waiting_since is None
        assert state.fail_reason is None
        assert isinstance(state.plan_state, PlanState)
        assert isinstance(state.content_state, ContentState)
        assert isinstance(state.review_state, ReviewState)
        assert isinstance(state.reflection_state, ReflectionState)

    def test_is_iteration_exceeded_false(self):
        """未超过最大迭代次数"""
        state = AgentState(max_iterations=3, current_iteration=2)
        assert not state.is_iteration_exceeded()

    def test_is_iteration_exceeded_equal(self):
        """达到最大迭代次数（等于时也超限）"""
        state = AgentState(max_iterations=3, current_iteration=3)
        assert state.is_iteration_exceeded()

    def test_is_iteration_exceeded_over(self):
        """超过最大迭代次数"""
        state = AgentState(max_iterations=3, current_iteration=5)
        assert state.is_iteration_exceeded()

    def test_is_confirmation_not_timed_out_no_waiting(self):
        """未设置 waiting_since 时不超时"""
        state = AgentState(waiting_since=None)
        assert not state.is_confirmation_timed_out()

    def test_is_confirmation_not_timed_out_within_limit(self):
        """在超时范围内"""
        state = AgentState(
            waiting_since=datetime.now(timezone.utc) - timedelta(minutes=10),
            confirmation_timeout_minutes=30,
        )
        assert not state.is_confirmation_timed_out()

    def test_is_confirmation_timed_out_exceeded(self):
        """超过超时时间"""
        state = AgentState(
            waiting_since=datetime.now(timezone.utc) - timedelta(minutes=31),
            confirmation_timeout_minutes=30,
        )
        assert state.is_confirmation_timed_out()

    def test_is_confirmation_timed_out_exact_boundary(self):
        """在超时边界上（刚好30分钟，不超时）"""
        state = AgentState(
            waiting_since=datetime.now(timezone.utc) - timedelta(minutes=29, seconds=59),
            confirmation_timeout_minutes=30,
        )
        assert not state.is_confirmation_timed_out()

    def test_is_reflection_converged_empty(self):
        """updated_plan 或 previous 为空时不收敛"""
        state = AgentState(convergence_threshold=0.9)
        state.reflection_state.updated_plan = ""
        state.reflection_state.previous_rejected_plan = ""
        assert not state.is_reflection_converged()

    def test_is_reflection_converged_high_similarity(self):
        """高相似度 → 收敛"""
        state = AgentState(convergence_threshold=0.9)
        state.reflection_state.updated_plan = "目标受众：年轻人，核心卖点：性价比，风格：活泼"
        state.reflection_state.previous_rejected_plan = "目标受众：年轻人，核心卖点：性价比极高，风格：活泼"
        # 这2句的相似度应该 > 0.9
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(
            None,
            state.reflection_state.updated_plan,
            state.reflection_state.previous_rejected_plan,
        ).ratio()
        # 确认相似度确实 >= 0.9
        assert ratio >= 0.9, f"Expected ratio >= 0.9, got {ratio}"
        assert state.is_reflection_converged()

    def test_is_reflection_converged_low_similarity(self):
        """低相似度 → 不收敛"""
        state = AgentState(convergence_threshold=0.9)
        state.reflection_state.updated_plan = (
            "目标受众：年轻人，核心卖点：性价比，风格：活泼"
        )
        state.reflection_state.previous_rejected_plan = (
            "目标受众：中老年人，核心卖点：健康养生，风格：专业严谨医学术语"
        )
        assert not state.is_reflection_converged()

    def test_is_reflection_converged_none_previous(self):
        """previous_rejected_plan 为 None 时不收敛"""
        state = AgentState(convergence_threshold=0.9)
        state.reflection_state.updated_plan = "test plan"
        state.reflection_state.previous_rejected_plan = None
        assert not state.is_reflection_converged()

    def test_get_current_result_planning(self):
        """planning 步骤返回 plan"""
        state = AgentState(step="planning")
        state.plan_state.plan = "策划方案内容"
        assert state.get_current_result() == "策划方案内容"

    def test_get_current_result_generating(self):
        """generating 步骤返回 content"""
        state = AgentState(step="generating")
        state.content_state.content = "营销文案内容"
        assert state.get_current_result() == "营销文案内容"

    def test_get_current_result_reviewing(self):
        """reviewing 步骤返回 result"""
        state = AgentState(step="reviewing")
        state.review_state.result = "审核结果"
        assert state.get_current_result() == "审核结果"

    def test_get_current_result_default(self):
        """未知步骤返回空字符串"""
        state = AgentState(step="unknown")
        assert state.get_current_result() == ""

    def test_state_serialization(self):
        """测试序列化/反序列化"""
        state = AgentState(
            user_input="测试选题",
            topic_id=1,
            step="planning",
            status="running",
        )
        state.plan_state.plan = "策划方案"

        # 序列化
        json_str = state.model_dump_json()
        assert "测试选题" in json_str
        assert "策划方案" in json_str

        # 反序列化
        restored = AgentState.model_validate_json(json_str)
        assert restored.user_input == "测试选题"
        assert restored.plan_state.plan == "策划方案"
        assert restored.step == "planning"
        assert restored.status == "running"


class TestReflectionState:

    def test_default_values(self):
        state = ReflectionState()
        assert state.analysis is None
        assert state.updated_plan is None
        assert state.previous_rejected_plan is None
        assert state.iteration_count == 0

    def test_can_store_previous_plan(self):
        """测试 previous_rejected_plan 字段存储"""
        state = ReflectionState(
            analysis="分析内容",
            updated_plan="新方案v2",
            previous_rejected_plan="旧方案v1",
            iteration_count=2,
        )
        assert state.previous_rejected_plan == "旧方案v1"
        assert state.updated_plan == "新方案v2"
        assert state.iteration_count == 2


class TestToolCallRecord:

    def test_create_record(self):
        record = ToolCallRecord(
            tool_name="check_sensitive",
            args={"content": "测试内容"},
            result={"is_pass": True},
            timestamp="2026-06-21T10:00:00",
        )
        assert record.tool_name == "check_sensitive"
        assert record.result == {"is_pass": True}