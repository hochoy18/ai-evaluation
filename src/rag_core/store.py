"""Chroma 封装 + manifest 读写(架构 §6、§4.5)。

collection:rag_corpus,cosine HNSW;id 规则 {doc_id}:{chunk_index}。
manifest:embedding_model / embedding_dim / index_version / 每文件差量状态。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from rag_core.config import Config

COLLECTION_NAME = "rag_corpus"
INDEX_VERSION = 1  # 索引布局版本;与 manifest 不一致 → 拒查提示 --rebuild


class IndexMismatchError(RuntimeError):
    """manifest 与当前配置不一致(或索引缺失):拒绝查询,提示 --rebuild。"""


@dataclass
class FileEntry:
    content_hash: str
    doc_id: str
    chunk_ids: list[str]
    mtime: float


@dataclass
class Manifest:
    embedding_model: str
    embedding_dim: int
    index_version: int
    files: dict[str, FileEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "index_version": self.index_version,
            "files": {
                rel: {
                    "content_hash": e.content_hash,
                    "doc_id": e.doc_id,
                    "chunk_ids": e.chunk_ids,
                    "mtime": e.mtime,
                }
                for rel, e in self.files.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Manifest":
        return cls(
            embedding_model=raw["embedding_model"],
            embedding_dim=int(raw["embedding_dim"]),
            index_version=int(raw["index_version"]),
            files={
                rel: FileEntry(
                    content_hash=e["content_hash"],
                    doc_id=e["doc_id"],
                    chunk_ids=list(e["chunk_ids"]),
                    mtime=float(e["mtime"]),
                )
                for rel, e in raw.get("files", {}).items()
            },
        )


def load_manifest(path: Path) -> Manifest | None:
    if not path.exists():
        return None
    return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _client(config: Config) -> ClientAPI:
    return chromadb.PersistentClient(
        path=str(config.index_dir),
        settings=Settings(anonymized_telemetry=False),  # hermetic:零网络
    )


def open_collection(config: Config) -> Collection:
    """建索引侧入口:cosine collection,不存在则创建。"""
    return _client(config).get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def validate_manifest(manifest: Manifest | None, config: Config) -> None:
    """查询前校验(架构 §4.5):模型/维度/索引版本不一致 → 拒查提示 --rebuild。

    防「换模型后新旧向量混检索」的静默错误。
    """
    if manifest is None:
        raise IndexMismatchError("索引不存在:先运行 python -m rag_core.ingest 建索引")
    if (
        manifest.embedding_model != config.embedding_model
        or manifest.embedding_dim != config.embedding_dim
        or manifest.index_version != INDEX_VERSION
    ):
        raise IndexMismatchError(
            f"manifest(model={manifest.embedding_model}, dim={manifest.embedding_dim}, "
            f"version={manifest.index_version})与当前配置(model={config.embedding_model}, "
            f"dim={config.embedding_dim}, version={INDEX_VERSION})不一致:"
            "拒绝查询,请运行 python -m rag_core.ingest --rebuild"
        )


def open_query_collection(config: Config) -> Collection:
    """查询侧入口:先过 manifest 门禁再开 collection(门面接线随 vector 后端票)。"""
    validate_manifest(load_manifest(config.manifest_file), config)
    return open_collection(config)


def reset_collection(config: Config) -> Collection:
    """--rebuild:清空 collection 重建(杂散向量与旧模型向量一并清掉)。"""
    client = _client(config)
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
