"""
P0 评测门禁:关键词医疗版门面 × 四指标 × bailian_judge(qwen3.7-plus, temp 0.0)。

- 真题经 RAG_GOLDENS_FILE env 读取(默认 data/rag_goldens_medical.json),
  医疗/业务切集零门禁代码改动(架构 §10.1);
- schema 前置校验在 collection 期生效:格式错误先于任何 judge 调用报错,
  不让 judge 静默给 0;
- 门禁集 = single_hop + cross_doc;out_of_domain 剔出四指标门禁
  (拒答句使 ContextualRecall 必然趋 0,DeepEval 4.3.4 源码已核实);
- 阈值为初值(架构 §10.4),基线分布建立后校准;
- 门面三签名与装饰器位置冻结(ADR 0001),评测侧只 import 门面;
  检索上下文 = 按排名有序的 list[str],原样喂 retrieval_context。

运行(正典运行器,产出 inspect 所需 test_run 快照):
    uv run deepeval test run src/deepeval_cases/test_rag.py
失败注入(钩子在门面层,经 sys.argv 生效,见 conftest.py):
    uv run deepeval test run src/deepeval_cases/test_rag.py --bad-retrieval
    uv run deepeval test run src/deepeval_cases/test_rag.py --hallucinate
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval import assert_test  # type: ignore[attr-defined]  # deepeval 运行时懒加载
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ai_evaluation.goldens import load_goldens
from ai_evaluation.models import bailian_judge
from deepeval_cases.rag_app import rag_app

load_dotenv()

# ---------- 真题资产(env 切集,零代码改动) ----------

REPO_ROOT = Path(__file__).resolve().parents[2]
_goldens_file = Path(os.getenv("RAG_GOLDENS_FILE", "data/rag_goldens_medical.json"))
if not _goldens_file.is_absolute():
    _goldens_file = REPO_ROOT / _goldens_file

# 前置校验:格式错误在 collection 期抛出,先于任何指标/judge 调用
GOLDENS = load_goldens(_goldens_file)
# 门禁集 = single_hop + cross_doc;out_of_domain 剔出四指标门禁(留拒答集,P3)
GATE_GOLDENS = [g for g in GOLDENS if g["difficulty"] != "out_of_domain"]

# ---------- 裁判与阈值(初值,架构 §10.4;基线分布后校准) ----------

JUDGE_MODEL = "qwen3.7-plus"  # token-plan 实测可用(架构原稿 qwen-plus 在该端点 404,见 issue #2)
judge = bailian_judge(JUDGE_MODEL, temperature=0.0)

FAITHFULNESS_THRESHOLD = 0.8  # 初值
ANSWER_RELEVANCY_THRESHOLD = 0.8  # 初值
CONTEXTUAL_PRECISION_THRESHOLD = 0.7  # 初值
CONTEXTUAL_RECALL_THRESHOLD = 0.7  # 初值


@pytest.mark.parametrize("golden", GATE_GOLDENS, ids=[g["id"] for g in GATE_GOLDENS])
def test_rag_gate(golden: dict):
    answer, contexts = rag_app(golden["question"])
    test_case = LLMTestCase(
        input=golden["question"],
        actual_output=answer,
        expected_output=golden["expected_output"],
        retrieval_context=contexts,  # 有序 list[str],排名即数据
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge),
            AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model=judge),
            ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD, model=judge),
            ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD, model=judge),
        ],
    )