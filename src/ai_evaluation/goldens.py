"""真题(golden)资产加载与 schema 前置校验(架构 §10.1)。

门禁模块在 collection 期调用 ``load_goldens``:格式错误先于任何 judge
调用抛出 ``GoldenSchemaError``,不让 judge 静默给 0。校验与语料无关
(不检查 expected_doc_ids 是否真实存在——那是检索指标的事)。
"""

import json
from pathlib import Path
from typing import Any, TypedDict


class Golden(TypedDict):
    id: str
    question: str
    expected_output: str
    expected_doc_ids: list[str]
    difficulty: str  # single_hop | cross_doc | out_of_domain
    source: str  # human | synthetic
    notes: str


class GoldenSchemaError(ValueError):
    """真题集不符合 schema:在跑指标前报错。"""


def load_goldens(path: str | Path) -> list[Golden]:
    """读取并校验真题集;任何格式错误抛 GoldenSchemaError。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate(raw)


REQUIRED_FIELDS = (
    "id",
    "question",
    "expected_output",
    "expected_doc_ids",
    "difficulty",
    "source",
    "notes",
)
DIFFICULTIES = ("single_hop", "cross_doc", "out_of_domain")
SOURCES = ("human", "synthetic")


def _validate(raw: Any) -> list[Golden]:
    if not isinstance(raw, list):
        raise GoldenSchemaError("真题集必须是 JSON 数组")
    goldens: list[Golden] = []
    seen_ids: set[str] = set()
    for i, g in enumerate(raw):
        if not isinstance(g, dict):
            raise GoldenSchemaError(f"真题 #{i}: 必须是 JSON 对象")
        label = g.get("id") or f"#{i}"
        for field in REQUIRED_FIELDS:
            if field not in g:
                raise GoldenSchemaError(f"真题 {label}: 缺少字段 {field}")
        for field in ("id", "question", "expected_output", "difficulty", "source", "notes"):
            if not isinstance(g[field], str) or not g[field].strip():
                raise GoldenSchemaError(f"真题 {label}: {field} 必须是非空字符串")
        doc_ids = g["expected_doc_ids"]
        if not isinstance(doc_ids, list) or not all(
            isinstance(d, str) and d.strip() for d in doc_ids
        ):
            raise GoldenSchemaError(f"真题 {label}: expected_doc_ids 必须是字符串数组")
        if g["difficulty"] not in DIFFICULTIES:
            raise GoldenSchemaError(
                f"真题 {label}: difficulty 必须是 {'/'.join(DIFFICULTIES)},实际 {g['difficulty']!r}"
            )
        if g["source"] not in SOURCES:
            raise GoldenSchemaError(
                f"真题 {label}: source 必须是 {'/'.join(SOURCES)},实际 {g['source']!r}"
            )
        if g["id"] in seen_ids:
            raise GoldenSchemaError(f"真题 {label}: id 重复")
        seen_ids.add(g["id"])
        if g["difficulty"] == "cross_doc" and len(doc_ids) < 2:
            raise GoldenSchemaError(
                f"真题 {label}: cross_doc 必须 ≥2 个 expected_doc_ids,实际 {len(doc_ids)} 个"
            )
        if g["difficulty"] == "out_of_domain" and doc_ids:
            raise GoldenSchemaError(
                f"真题 {label}: out_of_domain 的 expected_doc_ids 必须为空,实际 {doc_ids}"
            )
        goldens.append(
            Golden(
                id=g["id"],
                question=g["question"],
                expected_output=g["expected_output"],
                expected_doc_ids=list(doc_ids),
                difficulty=g["difficulty"],
                source=g["source"],
                notes=g["notes"],
            )
        )
    return goldens