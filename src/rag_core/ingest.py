"""建索引 CLI(架构 §4.5/§4.6):语料 → Chunk → 向量索引,增量更新 + 真题保鲜。

    uv run python -m rag_core.ingest            # 增量:按 manifest 差量更新
    uv run python -m rag_core.ingest --rebuild  # 全量:清空 collection 重建
    uv run python -m rag_core.ingest --dry-run  # 只报告三集合,不动索引

不在查询链路上;门面一字不动(ADR 0001)。Embedder 经协议替换位注入
(hermetic 测试),默认按 EMBEDDING_BACKEND env 构造。
"""

import argparse
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai_evaluation.goldens import load_goldens
from rag_core.chunking import chunk_document, doc_id_of
from rag_core.config import Config
from rag_core.embeddings import Embedder, make_embedder
from rag_core.store import (
    INDEX_VERSION,
    FileEntry,
    Manifest,
    IndexMismatchError,
    load_manifest,
    open_collection,
    reset_collection,
    save_manifest,
    validate_manifest,
)


def _freshness(config: Config, affected_paths: list[str]) -> None:
    """真题保鲜(架构 §4.6/§10.1):变更/删除源文档相对路径 ∩ expected_doc_ids → 人工 review 清单。

    永不自动失效、永不改真题文件——门禁构成不静默变化。
    expected_doc_ids = 语料内相对路径(§10.1),与 sha1 doc_id 无关。
    """
    if not affected_paths:
        print("[freshness] 无变更/删除源文档,真题不受影响")
        return
    try:
        goldens = load_goldens(config.goldens_file)
    except (OSError, ValueError) as e:  # 缺失/格式错:跳过保鲜,不挂建索引
        print(f"[freshness] 真题集不可用({e}),跳过保鲜清单")
        return
    hits = []
    for g in goldens:
        inter = sorted(set(g["expected_doc_ids"]) & set(affected_paths))
        if inter:
            hits.append((g, inter))
    if not hits:
        print("[freshness] 变更/删除源文档与真题 expected_doc_ids 交集为空,无受影响真题")
        return
    print(f"[freshness] 受影响真题 {len(hits)} 条(人工 review;真题永不自动失效,门禁构成不静默变化):")
    for g, inter in hits:
        print(f"  {g['id']} | {g['question']} | 受影响源文档: {', '.join(inter)}")


def _scan(corpus_dir: Path) -> dict[str, Path]:
    """语料 = 目录下全部 md(相对路径为键,POSIX 分隔,排序保证确定性)。"""
    return {
        p.relative_to(corpus_dir).as_posix(): p
        for p in sorted(corpus_dir.rglob("*.md"))
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None, *, embedder: Embedder | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_core.ingest", description="语料 → Chunk → 向量索引(增量)"
    )
    parser.add_argument("--rebuild", action="store_true", help="全量:清空 collection 重建")
    parser.add_argument("--dry-run", action="store_true", help="只报告三集合,不动索引")
    args = parser.parse_args(argv)

    load_dotenv()
    config = Config.from_env()

    files = _scan(config.corpus_dir)
    manifest = None if args.rebuild else load_manifest(config.manifest_file)
    if args.rebuild:
        print("[rebuild] 清空 collection 与 manifest,全量重建")
    elif manifest is not None:
        try:
            validate_manifest(manifest, config)
        except IndexMismatchError as e:
            print(f"[ingest] {e}")
            return 2  # 拒绝混嵌:换模型必须 --rebuild,不做静默增量
    old_files = manifest.files if manifest else {}
    hashes = {rel: _file_hash(p) for rel, p in files.items()}
    added = sorted(set(files) - set(old_files))
    changed = sorted(
        rel for rel in set(files) & set(old_files) if hashes[rel] != old_files[rel].content_hash
    )
    deleted = sorted(set(old_files) - set(files))
    skipped = sorted(set(files) & set(old_files) - set(changed))

    print(f"[ingest] corpus={config.corpus_dir} files={len(files)}")
    print(
        f"[ingest] added={len(added)} changed={len(changed)} "
        f"deleted={len(deleted)} skipped={len(skipped)}"
    )
    for rel in added:
        print(f"  + {rel}")
    for rel in changed:
        print(f"  ~ {rel}")
    for rel in deleted:
        print(f"  - {rel}")

    affected_paths = sorted(set(changed + deleted))
    if args.dry_run:
        print("[dry-run] 只报告三集合,索引与 manifest 未改动")
        _freshness(config, affected_paths)
        return 0

    col = reset_collection(config) if args.rebuild else open_collection(config)
    embedder = embedder if embedder is not None else make_embedder(config)
    new_manifest = Manifest(
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        index_version=INDEX_VERSION,
        files={rel: old_files[rel] for rel in skipped},
    )
    embedded_chunks = 0
    for rel in added + changed:
        path = files[rel]
        chunks = chunk_document(path.read_text(encoding="utf-8"), rel, config)
        embedded_chunks += len(chunks)
        doc_id = doc_id_of(rel)
        if rel in changed:
            col.delete(where={"doc_id": doc_id})  # 先删后插:chunk 数变少时不留残尾
        ids = [f"{doc_id}:{c.metadata['chunk_index']}" for c in chunks]
        if chunks:
            vectors = embedder.embed([c.text for c in chunks])
            col.upsert(
                ids=ids,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
                embeddings=vectors,  # type: ignore[arg-type]  # stub 窄于运行时(list[list[float]] 实际接受)
            )
        new_manifest.files[rel] = FileEntry(
            content_hash=hashes[rel],
            doc_id=doc_id,
            chunk_ids=ids,
            mtime=path.stat().st_mtime,
        )
    for rel in deleted:
        col.delete(where={"doc_id": old_files[rel].doc_id})  # 删除文件:按 doc_id 清理

    usage = getattr(embedder, "usage", None)
    if usage is not None:
        cost = usage.total_tokens / 1_000_000 * config.embedding_cost_per_million_tokens
        print(
            f"[embedding] {embedded_chunks} chunks, {usage.requests} 次请求"
            f"(batch≤{config.embedding_batch})"
        )
        print(
            f"[embedding] tokens={usage.total_tokens} 成本≈¥{cost:.4f}"
            f"({config.embedding_cost_per_million_tokens}元/百万token)"
        )

    save_manifest(config.manifest_file, new_manifest)
    print(
        f"[manifest] written {config.manifest_file} "
        f"(model={config.embedding_model} dim={config.embedding_dim})"
    )
    _freshness(config, affected_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
