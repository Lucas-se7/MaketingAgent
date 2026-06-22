"""
敏感词检测工具
支持关键词匹配，MVP 阶段使用
"""
import re
from typing import Literal

# MVP 敏感词库（示例）
SENSITIVE_WORDS = {
    "政治敏感": ["敏感词1", "敏感词2"],
    "违规广告": ["违规词1", "极限词1"],
    "色情低俗": ["低俗词1"],
}

# 极限词/违禁词常用模式
FORBIDDEN_PATTERNS = [
    r"第一[名次]",
    r"最佳|最优",
    r"国家级|世界级",
    r"100%|全网",
    r"无效退款",
]


class SensitiveResult:
    """敏感词检测结果"""

    def __init__(
        self,
        is_pass: bool,
        detected_type: str | None = None,
        detected_words: list[str] | None = None,
        message: str = ""
    ):
        self.is_pass = is_pass
        self.detected_type = detected_type
        self.detected_words = detected_words or []
        self.message = message

    def to_dict(self) -> dict:
        return {
            "is_pass": self.is_pass,
            "detected_type": self.detected_type,
            "detected_words": self.detected_words,
            "message": self.message
        }


def check_sensitive(content: str) -> SensitiveResult:
    """
    检测内容是否包含敏感词

    Args:
        content: 待检测内容

    Returns:
        SensitiveResult: 检测结果
    """
    if not content:
        return SensitiveResult(is_pass=True, message="内容为空，跳过检测")

    # 1. 检查违禁词模式
    for pattern in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            return SensitiveResult(
                is_pass=False,
                detected_type="违禁词模式",
                detected_words=matches,
                message=f"检测到违禁词模式: {matches[0]}"
            )

    # 2. 检查敏感词库
    for category, words in SENSITIVE_WORDS.items():
        for word in words:
            if word in content:
                return SensitiveResult(
                    is_pass=False,
                    detected_type=category,
                    detected_words=[word],
                    message=f"检测到{category}词: {word}"
                )

    return SensitiveResult(is_pass=True, message="未检测到敏感词")


def check_sensitive_with_filter(content: str) -> tuple[str, SensitiveResult]:
    """
    检测并过滤内容中的敏感词

    Returns:
        (filtered_content, result)
    """
    result = check_sensitive(content)

    if result.is_pass:
        return content, result

    # 过滤掉敏感词（替换为 ***）
    filtered = content
    for word in result.detected_words:
        filtered = filtered.replace(word, "***")

    result.message += "（已过滤）"
    return filtered, result
