"""Embedder 协议 + 实现(架构 §5、§9.2)。

协议替换位:ingest CLI 接受任意 Embedder(hermetic 测试注入 FakeEmbedder);
EMBEDDING_BACKEND env 分发 bailian(默认,按量计费 key)/ fake(确定性假向量,
零网络零计费,仅供 hermetic 测试与无 key 演示,检索质量无意义)。
"""

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from rag_core.config import API_BATCH_LIMIT, Config, clamp_batch


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class EmbedUsage:
    """ingestion 日志用:请求数 + token 用量(成本估算在 CLI 侧算)。"""

    requests: int = 0
    total_tokens: int = 0


class FakeEmbedder:
    """确定性假向量:同文本 → 同向量,异文本 → 异向量;内部同样按 batch ≤10 分批。"""

    def __init__(self, dim: int, batch: int = API_BATCH_LIMIT):
        self.dim = dim
        self.batch = clamp_batch(batch)
        self.usage = EmbedUsage()
        self.embedded: list[str] = []  # 缝上观察点:验证「只重嵌变更文件」

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            self.usage.requests += 1
            for text in texts[start : start + self.batch]:
                self.embedded.append(text)
                seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
                rng = random.Random(seed)
                vec = [rng.uniform(-1, 1) for _ in range(self.dim)]
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                out.append([x / norm for x in vec])
        return out


class BailianEmbedder:
    """百炼 text-embedding-v4(按量计费 DASHSCOPE_API_KEY;token-plan key 实测不可用)。

    batch ≤10 硬上限(官方 v3/v4 一致,附录 E 实测);失败重试 + 429 指数退避;
    usage 累计请求数与 token 用量,供 ingestion 日志成本估算。
    """

    def __init__(self, config: Config):
        if not config.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置:embedding 走按量计费 key(架构 §5.1);"
                "零计费演示可用 EMBEDDING_BACKEND=fake"
            )
        self.model = config.embedding_model
        self.dim = config.embedding_dim
        self.batch = clamp_batch(config.embedding_batch)  # >10 永不发出
        self.max_retries = config.embedding_max_retries
        self.usage = EmbedUsage()
        self._client = OpenAI(
            base_url=config.embedding_base_url,
            api_key=config.dashscope_api_key,
            max_retries=0,  # 重试自己管:429 单独退避
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            resp = self._call_with_retry(texts[start : start + self.batch])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
            self.usage.requests += 1
            if resp.usage is not None:
                self.usage.total_tokens += resp.usage.total_tokens
        return out

    def _call_with_retry(self, batch: list[str]):
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dim
                )
            except RateLimitError as e:
                if attempt >= self.max_retries:
                    raise
                # 429 单独处理(架构 §5.2):优先服务端 Retry-After,无则指数退避
                time.sleep(self._retry_after_seconds(e) or delay)
                delay *= 2
            except (APIConnectionError, APITimeoutError, InternalServerError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(delay)  # 瞬时故障:指数退避
                delay *= 2
        raise RuntimeError("unreachable")

    @staticmethod
    def _retry_after_seconds(e: RateLimitError) -> float | None:
        raw = e.response.headers.get("retry-after")
        try:
            return float(raw) if raw is not None else None
        except ValueError:  # HTTP-date 等不可解析格式 → 退回指数退避
            return None


def make_embedder(config: Config) -> Embedder:
    if config.embedding_backend == "fake":
        return FakeEmbedder(dim=config.embedding_dim, batch=config.embedding_batch)
    if config.embedding_backend == "bailian":
        return BailianEmbedder(config)
    raise ValueError(f"EMBEDDING_BACKEND={config.embedding_backend!r} 无效(可选 bailian/fake)")
