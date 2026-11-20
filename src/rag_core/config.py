"""配置集中地(架构 §3.2):全部参数于此,env 覆盖,不硬编码。

初值来源:架构 §4.3(chunk 字符计量)、§5.2(embedding)、§12.1(env 清单)。
chunk 参数均为初值,P1 在 golden set 上扫参校准。
"""

import os
from dataclasses import dataclass
from pathlib import Path

# text-embedding-v3/v4 官方批量上限(实测 batch=11 返回 400,架构附录 E)
API_BATCH_LIMIT = 10

# 双 key 双 endpoint(架构 §5.1):embedding 走按量计费,独立于生成端 token-plan
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def clamp_batch(batch: int) -> int:
    """batch >10 永不发出:配置层与实现层共用这一道夹取。"""
    return max(1, min(batch, API_BATCH_LIMIT))


@dataclass(frozen=True)
class Config:
    corpus_dir: Path
    index_dir: Path
    manifest_file: Path
    goldens_file: Path
    embedding_backend: str  # bailian(默认,按量计费)| fake(确定性假向量,零网络)
    embedding_model: str
    embedding_dim: int
    embedding_batch: int
    dashscope_api_key: str
    embedding_cost_per_million_tokens: float  # 初值 0.5 元/百万 token(架构 §5.1)
    embedding_max_retries: int
    chunk_target: int
    chunk_max: int
    chunk_min: int
    chunk_overlap: int
    embedding_base_url: str = DASHSCOPE_BASE_URL

    @classmethod
    def from_env(cls) -> "Config":
        batch = int(os.getenv("EMBEDDING_BATCH", str(API_BATCH_LIMIT)))
        return cls(
            corpus_dir=Path(os.getenv("RAG_CORPUS_DIR", "./docs")),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", "./data/rag_index")),
            manifest_file=Path(os.getenv("RAG_MANIFEST_FILE", "./data/rag_manifest.json")),
            goldens_file=Path(os.getenv("RAG_GOLDENS_FILE", "./data/rag_goldens.json")),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "bailian"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            # batch >10 永不发出:配置层先夹一道,实现层再夹一道
            embedding_batch=clamp_batch(batch),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", DASHSCOPE_BASE_URL),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            embedding_cost_per_million_tokens=float(
                os.getenv("EMBEDDING_COST_PER_MILLION_TOKENS", "0.5")
            ),
            embedding_max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "5")),
            chunk_target=int(os.getenv("RAG_CHUNK_TARGET", "700")),
            chunk_max=int(os.getenv("RAG_CHUNK_MAX", "1100")),
            chunk_min=int(os.getenv("RAG_CHUNK_MIN", "140")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        )
