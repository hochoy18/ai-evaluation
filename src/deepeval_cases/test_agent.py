"""
Agent 评测 pytest 形态 —— 学习路径 §4 第 4 步(CI 门禁形态)。

与脚本形态(agent_tools.py)的对照:
    evals_iterator 循环   →  @pytest.mark.parametrize("golden", goldens)
    循环内自动断言        →  assert_test(golden=golden, metrics=[...])
    uv run python xxx.py →  uv run deepeval test run src/deepeval_cases/test_agent.py

两个关键机制(v4.3.4 源码核实):
- assert_test(golden=...) 从 current_trace_context 读活 trace,
  必须用 deepeval test run(插件让 trace 在整个测试体内保持存活),
  原生 pytest 会报 "No active trace found";
- call_llm 里动态挂到决策 span 的组件指标(ToolCorrectness/ArgumentCorrectness)
  随 trace 一起被自动评测,不用在这里重复声明。

agent 复用 agent_tools.travel_agent(已 @observe 插桩)。
import 时实验开关自动为 False → CI 只测基线路径。

运行:
    uv run deepeval test run src/deepeval_cases/test_agent.py
全量门禁(单轮+多轮+agent):
    uv run deepeval test run src/deepeval_cases/
"""

import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import StepEfficiencyMetric, TaskCompletionMetric

from deepeval_cases.agent_tools import judge_model, travel_agent

goldens = [
    Golden(input="订明天从纽约到洛杉矶最便宜的航班"),
]


@pytest.mark.parametrize("golden", goldens)
def test_flight_agent(golden: Golden):
    travel_agent(golden.input)  # trace 由 @observe 自动捕获
    assert_test(
        golden=golden,  # 传 golden 不是 test_case:DeepEval 自动组装轨迹用例
        metrics=[
            TaskCompletionMetric(threshold=0.7, model=judge_model),
            StepEfficiencyMetric(threshold=0.7, model=judge_model),
        ],
    )
