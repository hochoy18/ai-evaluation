"""
真题 schema 前置校验 —— seam 1(门禁模块 × 运行器 × env)的 hermetic 切片。

只断言 seam 上的外部行为:子进程跑门禁收集(pytest --collect-only),
观察「格式错误在跑指标前报错」与「门禁集构成」。校验在 collection 期
生效,收集失败 = judge 从未被调用,零网络零计费。

运行(裸 pytest 即可,本文件不碰 LLM):
    uv run pytest src/deepeval_cases/test_golden_schema.py -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_FILE = REPO_ROOT / "src" / "deepeval_cases" / "test_rag.py"


def make_golden(**overrides) -> dict:
    base = {
        "id": "g001",
        "question": "腋下体温37.5℃算发烧吗?",
        "expected_output": "腋下体温≥37.3℃即属发烧,37.5℃属低烧。",
        "expected_doc_ids": ["fever"],
        "difficulty": "single_hop",
        "source": "synthetic",
        "notes": "schema 测试 fixture",
    }
    base.update(overrides)
    return base


def write_goldens(tmp_path: Path, goldens: object) -> Path:
    path = tmp_path / "goldens.json"
    path.write_text(json.dumps(goldens, ensure_ascii=False), encoding="utf-8")
    return path


def collect_gate(goldens_file: Path) -> subprocess.CompletedProcess:
    """经 RAG_GOLDENS_FILE env 切集,子进程收集门禁(不执行测试体)。"""
    env = {
        **os.environ,
        "RAG_GOLDENS_FILE": str(goldens_file),
        "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(GATE_FILE), "--collect-only", "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cross_doc_with_single_expected_doc_rejected_at_collection(tmp_path):
    """cross_doc 期望源文档 <2:收集期报 schema 错,先于任何 judge 调用。"""
    path = write_goldens(
        tmp_path, [make_golden(difficulty="cross_doc", expected_doc_ids=["fever"])]
    )
    proc = collect_gate(path)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"应收集失败,实际输出:\n{output}"
    assert "cross_doc" in output


def test_out_of_domain_with_expected_docs_rejected_at_collection(tmp_path):
    """out_of_domain 期望非空:收集期报 schema 错(域外题不可能有期望源文档)。"""
    path = write_goldens(
        tmp_path,
        [make_golden(difficulty="out_of_domain", expected_doc_ids=["fever"])],
    )
    proc = collect_gate(path)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"应收集失败,实际输出:\n{output}"
    assert "out_of_domain" in output


def test_gate_set_excludes_out_of_domain(tmp_path):
    """门禁集 = single_hop + cross_doc;out_of_domain 不进四指标门禁。"""
    path = write_goldens(
        tmp_path,
        [
            make_golden(id="g001"),
            make_golden(
                id="g002",
                difficulty="cross_doc",
                expected_doc_ids=["fever", "medication"],
            ),
            make_golden(
                id="g003",
                question="明天北京天气怎么样?",
                expected_output="资料中没有相关信息。",
                difficulty="out_of_domain",
                expected_doc_ids=[],
            ),
        ],
    )
    proc = collect_gate(path)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"合法真题集应收集成功,实际输出:\n{output}"
    assert "test_rag_gate[g001]" in output
    assert "test_rag_gate[g002]" in output
    assert "g003" not in output


def test_env_switches_golden_set_without_code_change(tmp_path):
    """RAG_GOLDENS_FILE 切集:同一份门禁代码收集到另一套真题。"""
    other = write_goldens(tmp_path, [make_golden(id="biz001"), make_golden(id="biz002")])
    proc = collect_gate(other)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"合法真题集应收集成功,实际输出:\n{output}"
    assert "test_rag_gate[biz001]" in output
    assert "test_rag_gate[biz002]" in output
    assert "g001" not in output


@pytest.mark.parametrize(
    ("payload", "expected_hint"),
    [
        pytest.param(
            [{k: v for k, v in make_golden().items() if k != "notes"}],
            "notes",
            id="missing-required-field",
        ),
        pytest.param(
            [make_golden(difficulty="multi_hop")],
            "difficulty",
            id="bad-difficulty",
        ),
        pytest.param(
            [make_golden(expected_doc_ids="fever")],
            "expected_doc_ids",
            id="doc-ids-not-a-list",
        ),
        pytest.param(
            [make_golden(), make_golden()],
            "id",
            id="duplicate-id",
        ),
        pytest.param({"id": "g001"}, "数组", id="top-level-not-a-list"),
    ],
)
def test_malformed_goldens_rejected_at_collection(tmp_path, payload, expected_hint):
    """格式错误在跑指标前报错(collection 期),不让 judge 静默给 0。"""
    proc = collect_gate(write_goldens(tmp_path, payload))
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"应收集失败,实际输出:\n{output}"
    assert expected_hint in output