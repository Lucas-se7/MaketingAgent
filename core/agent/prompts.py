"""
Prompt 管理 - 支持数据库存储、输入清洗、Few-shot Examples
"""
import re
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- 注入内容安全配置 ---
MAX_INPUT_LENGTH = 4000      # 单个变量最大长度
MAX_PROMPT_LENGTH = 8000     # 最终 Prompt 最大长度

# 禁止的模式（提示词攻击特征）
FORBIDDEN_PATTERNS = [
    r"\((?:system|user|assistant)\s*:\s*",
    r"\{\{.*\}\}",
    r"<\|.*\|>",
    r"<\/.*>",
]


@dataclass
class PromptTemplate:
    """Prompt 模板"""
    name: str
    content: str
    version: int = 1


class PromptManager:
    """
    Prompt 管理器

    功能：
    - 从数据库加载 Prompt
    - 对外部输入进行清洗和截断
    - 组合 Few-shot Examples
    - 防止提示词注入攻击
    """

    def __init__(self):
        # MVP: 使用内存存储，生产环境应从数据库加载
        self._prompts: dict[str, PromptTemplate] = {}
        self._examples: dict[str, list[dict]] = {}
        self._init_default_prompts()

    def _init_default_prompts(self):
        """初始化默认 Prompt（MVP 直接内嵌）"""
        self._prompts = {
            "planner": PromptTemplate(
                name="planner",
                content="""你是一个营销策划专家。用户要推广的产品或主题是：

{user_input}

请根据上述信息，制定一个营销策划方案，包含：
1. 目标受众（谁会感兴趣）
2. 内容方向（讲什么故事）
3. 核心卖点（最重要的亮点）
4. 建议风格（专业/活泼/感性/幽默）

请用简洁清晰的方式输出策划方案。"""
            ),
            "generator": PromptTemplate(
                name="generator",
                content="""你是营销文案专家。请根据以下策划方案生成营销内容：

## 策划方案
{plan}

## 风格要求
{style}

## 模板
{template}

要求：
1. 突出核心卖点
2. 语言简洁有力
3. 符合平台调性
4. 直接输出文案，不要多余的话"""
            ),
            "reviewer": PromptTemplate(
                name="reviewer",
                content="""你是一个营销内容审核专家。请审核以下内容是否合格：

## 待审核内容
{content}

## 品牌知识参考
{knowledge}

请判断：
1. 内容是否符合品牌调性？
2. 是否有敏感词或违规内容？
3. 表达是否清晰有效？

输出格式：
- 通过/驳回
- 理由（如有）"""
            ),
            "reflection": PromptTemplate(
                name="reflection",
                content="""你是一个营销策划专家，需要分析内容被驳回的原因。

## 审核驳回原因
{review_result}

## 用户反馈
{feedback}

## 原策划方案
{original_plan}

请分析：
1. 驳回的核心问题是什么？
2. 需要如何调整策划方案？

输出格式：
- 问题分析
- 调整建议
- 更新后的策划方案（如果需要）"""
            ),
        }

        # 默认 Examples
        self._examples = {
            "generator": [
                {
                    "input": "新品发布：智能手表X",
                    "output": "🎉 智能手表X全新上市！\n\n⏰ 7天超长续航 | 💧 50米防水 | ❤️ 24h心率监测\n\n立即购买，享首发优惠！"
                },
                {
                    "input": "节日促销：双11狂欢",
                    "output": "🛒 双11狂欢节来啦！\n\n全场5折起 | 满减不封顶 | 限时秒杀\n\n错过等一年，点击立即抢购→"
                },
            ]
        }

    def sanitize_input(self, value: str, field_name: str) -> str:
        """
        对注入到 Prompt 的外部内容进行清洗和截断

        防御措施：
        1. 长度截断
        2. 去除提示词攻击特征
        3. 去除多余空白
        """
        if not value:
            return ""

        # 1. 长度截断
        if len(value) > MAX_INPUT_LENGTH:
            value = value[:MAX_INPUT_LENGTH]
            logger.warning(f"输入 {field_name} 超过 {MAX_INPUT_LENGTH} 字符，已截断")

        # 2. 去除潜在提示词攻击特征
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"输入 {field_name} 包含可疑模式，已过滤: {pattern}")
                value = re.sub(pattern, "", value, flags=re.IGNORECASE)

        # 3. 去除多余空白字符
        value = re.sub(r"\s+", " ", value).strip()

        return value

    def get(self, name: str) -> PromptTemplate:
        """获取 Prompt 模板"""
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' 不存在")
        return self._prompts[name]

    def get_examples(self, name: str) -> list[dict]:
        """获取 Prompt 的 Examples"""
        return self._examples.get(name, [])

    def get_with_examples(self, name: str) -> tuple[str, list[dict]]:
        """获取 Prompt 及关联的 Examples"""
        prompt = self.get(name)
        examples = self.get_examples(name)
        return prompt.content, examples

    def format_prompt(self, name: str, **kwargs: Any) -> str:
        """
        将 Prompt 模板与 Variables、Examples 组合成最终输入

        Args:
            name: Prompt 名称
            **kwargs: 变量键值对

        Returns:
            组合后的完整 Prompt
        """
        template, examples = self.get_with_examples(name)

        # 对所有输入进行清洗
        sanitized_kwargs = {
            k: self.sanitize_input(str(v), k)
            for k, v in kwargs.items()
        }

        # 填充变量
        try:
            prompt = template.format(**sanitized_kwargs)
        except KeyError as e:
            logger.error(f"Prompt 变量缺失: {e}")
            raise ValueError(f"Prompt '{name}' 缺少变量: {e}")

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


# 全局 Prompt 管理器实例
prompt_manager = PromptManager()
