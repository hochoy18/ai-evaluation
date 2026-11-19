"""百炼（阿里云 Bailian，OpenAI 兼容接口）裁判模型适配。

DeepEval 的 LLM-as-a-judge 指标默认用 OpenAI 的 gpt 模型。要换成百炼，
关键是三件事：base_url + api_key + 百炼模型名（qwen-*）。

两种用法：
1. 显式（推荐）：给每个指标传 ``model=bailian_judge("qwen-plus")``。
2. 全局：设环境变量 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL_NAME，
   让所有 ``model=None`` 的指标自动用百炼（见 .env.example）。
"""

import os

from deepeval.models import OpenAIModel

# 百炼 OpenAI 兼容 endpoint（可被环境变量 OPENAI_BASE_URL 覆盖）
BAILIAN_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)


def bailian_judge(model: str = "qwen3.7-plus", **kwargs) -> OpenAIModel:
    """返回一个指向百炼的 DeepEval 裁判模型。

    Args:
        model: 百炼模型名，如 qwen-plus / qwen-max / qwen-turbo（须为你的套餐支持的模型）。
        **kwargs: 透传给 OpenAIModel，如 temperature、generation_kwargs。

    api_key 优先读 BAILIAN_API_KEY，其次 OPENAI_API_KEY；都没有时由
    OpenAIModel 回退到 settings.OPENAI_API_KEY（即环境变量 OPENAI_API_KEY）。
    """
    api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    kwargs.setdefault("temperature", 0.0)
    return OpenAIModel(
        model=model,
        base_url=BAILIAN_BASE_URL,
        api_key=api_key,
        **kwargs,
    )