# DeepEval 快速入门详解（Getting Started / 5-min Quickstart）

> 原文：https://deepeval.com/docs/getting-started —《DeepEval 5-min Quickstart》
> 本文为详细中文讲解，代码保留原样（注释汉化），并结合本项目（uv 管理）给出落地写法。

---

## 0. 这一页是什么

这是 DeepEval 的「5 分钟快速上手」，目标是从安装到跑通第一个及格（passing）的评测。读完你应能：

1. 用「测试用例 + 指标 + `deepeval test run`」跑通第一个本地评测。
2. 用 tracing 评测 AI Agent 的**完整轨迹**，并诊断其**内部组件**。
3. 知道下一步去哪找数据集、集成、生产监控。

> 另一条路：如果你不想手写，DeepEval 提供一个 skill，可装进 Cursor / Claude Code / Codex / Windsurf 等编码工具，让编码 Agent 替你写测试套件、跑 `deepeval test run`、迭代失败项（见「Vibe Coder 5-min Quickstart」）。

---

## 1. 安装与环境

官方写法（在新建的虚拟环境里）：

```bash
pip install -U deepeval
```

**本项目用 uv**，等价写法（你已在 `pyproject.toml` 加了 `deepeval>=4.3.4`）：

```bash
uv add deepeval        # 已做
uv sync                # 同步到 .venv
```

要点：

- `deepeval` **接入 Pytest**，所以 `deepeval test run` 收集/运行评测文件的方式和 `pytest` 一样。
- 本快速入门**全部本地运行**。若还想把 trace 和测试运行发送到 Confident AI，再登录：

```bash
deepeval login
# CI 等非交互环境：
deepeval login --api-key ...
```

**环境变量**：DeepEval 在 import 时自动加载 env 文件。

- 优先级：已有进程环境变量 → `.env.local` → `.env`
- 关闭自动加载：设 `DEEPEVAL_DISABLE_DOTENV=1`

```bash
cp .env.example .env.local   # 然后编辑 .env.local（被 git 忽略）
```

---

## 2. 核心心智模型：测试用例 + 指标 + 运行

DeepEval 的一次评测 = **一个测试用例（test case）** 交给 **一个或多个指标（metric）** 打分，再用运行器跑起来。

### 2.1 测试用例（Test Case）

**单轮 `LLMTestCase`**：表示一次 LLM 应用交互的最小单元。

- 必填：`input`（模拟用户输入）、`actual_output`（你的应用实际产出）。
- 可选：`expected_output`（理想答案）。RAG 场景还会有 `context` / `retrieval_context`（本页未展开）。

**多轮 `ConversationalTestCase`**：表示多轮对话。

- `turns=[Turn(role="user"|"assistant", content=...)]`：实际发生的对话。
- 可选 `scenario`：对话发生的场景。
- `role` 区分终端用户与你的 LLM 应用，`content` 是用户输入或 LLM 输出。

### 2.2 指标（Metric）：两种「裁判」范式

所有指标分数范围 **0–1**，由 `threshold`（阈值）决定测试是否通过。

**① LLM-as-a-judge：`GEval`**（研究支撑的自定义指标，用 LLM 当裁判，接近人类判断精度）

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
)
```

- `criteria`：用自然语言描述的评判标准（这里是「基于 expected_output 判断 actual_output 是否正确」）。
- `evaluation_params`：告诉指标用测试用例的哪些字段来评。
- 需要 `OPENAI_API_KEY`（或配置任意其他 LLM）。
- **并非所有指标都需要 `expected_output`**。
- 多轮版本是 `ConversationalGEval`。

**② Jev-as-a-judge：`JevEval`**（结构化问题裁判，**无需 LLM key**，需 `TYPESAFE_API_KEY`）

```python
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul, Score
from deepeval.test_case import SingleTurnParams

correctness_metric = JevEval(
    name="Correctness",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    questions=[
        Noul("No fact in actual_output contradicts a fact in expected_output.", weight=2),
        Score(
            "How much of the detail in expected_output is present in actual_output?",
            levels=["Almost none", "Some", "Most", "All of it"],
        ),
    ],
    threshold=0.5,
)
```

- 用一组「问题」打分：`Noul` 是带 `weight`（权重）的断言项（如「actual_output 没有任何事实与 expected_output 矛盾」）；`Score` 是分级评分项（用 `levels` 定义档位）。
- 多轮版本是 `ConversationalJevEval`（配 `MultiTurnParams`）。
- 这是较新的范式，适合不想依赖 LLM 裁判、想要更结构化/可加权评分的场景。

> 版本提示：本页反映的是最新版 API（含 `JevEval`、`SingleTurnParams` 等）。你 `pyproject.toml` 写的是 `deepeval>=4.3.4`，会解析到当前最新版；若锁定在旧 4.x，个别符号名可能不同，以 `uv run python -c "import deepeval; print(deepeval.__version__)"` 实际版本为准。

### 2.3 运行与断言

```python
from deepeval import assert_test
assert_test(test_case, [correctness_metric])   # 单轮
# 多轮：assert_test(conversational_test_case, [professionalism_metric])
```

把上面放进 `test_example.py` 的 `def test_correctness():` 里，然后从项目根目录运行：

```bash
deepeval test run test_example.py
# 本项目 uv 写法：
uv run deepeval test run test_example.py
```

跑多次测试运行后，可并排比较测试用例来**捕捉回归**。

---

## 3. 第一个端到端评测（完整最小例）

```python
# test_example.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # 把这里替换成你的 LLM 应用的真实输出
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or CDC-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs.",
    )
    assert_test(test_case, [correctness_metric])
```

发生了什么：

- `input` 模拟用户输入；`actual_output` 是应用「应当输出」的占位。
- `expected_output` 是该输入的理想答案；`GEval` 按自定义 `criteria` 评判。
- 分数 0–1，`threshold=0.5` 决定通过与否。

---

## 4. 保存与查看结果

测试运行会**本地缓存**，可自选保存位置；登录后也会出现在 Confident AI。

```bash
# 本地 JSON：设置结果目录
export DEEPEVAL_RESULTS_FOLDER="./data"     # linux/mac
# set DEEPEVAL_RESULTS_FOLDER=.\data        # windows
```

- 本地存储（含 SQLite 替代方案）详见官方 env 文档。
- 平台查看：`deepeval view`（要求该测试运行已成功本地缓存；若报错，先 `deepeval login` 再重跑）。

---

## 5. 评测 AI Agent（轨迹 + 组件）

Agent 往往要多步才出结果。要评「是否完成任务」以及「怎么完成的」：先用 LLM tracing 插桩，再对完整轨迹打分。

### 5.1 构建数据集

```python
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[Golden(input="Plan a three-day trip to Paris")])
```

`Golden` 是「标准样例」（这里只有 input，代表一个代表性任务）。

### 5.2 插桩 + 轨迹评测

核心：用 `@observe` 包裹顶层函数，用 `update_current_trace(...)` 设置轨迹级测试用例字段；用 `dataset.evals_iterator(metrics=[TaskCompletionMetric()])` 循环，每个 golden 捕获一条完整 trace。

每个集成都提供 **Async（默认，并发跑 goldens）** 与 **Sync（一次一个，便于调试或受限流供应商）** 两种样例。

**手动插桩 — Async：**

```python
# main.py
import asyncio
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import TaskCompletionMetric
# ...（dataset 定义见上）

@observe()
async def my_ai_agent(query: str) -> str:
    answer = "..."  # 在这里 await 你的 LLM 调用
    update_current_trace(input=query, output=answer)
    return answer

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric()]):
    task = asyncio.create_task(my_ai_agent(golden.input))
    dataset.evaluate(task)
```

**手动插桩 — Sync：**

```python
from deepeval.evaluate import AsyncConfig
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import TaskCompletionMetric

@observe()
def my_ai_agent(query: str) -> str:
    answer = "..."  # 在这里调用你的 LLM
    update_current_trace(input=query, output=answer)
    return answer

for golden in dataset.evals_iterator(
    metrics=[TaskCompletionMetric()],
    async_config=AsyncConfig(run_async=False),
):
    my_ai_agent(golden.input)
```

运行：

```bash
python main.py        # 本项目：uv run python main.py
```

运行 recap：

- `evals_iterator()` 遍历数据集，每个 golden 捕获一条 trace。
- 集成适配器（或 `@observe`）创建 span，记录 Agent 内部步骤。
- `TaskCompletionMetric` 在 Agent 结束后分析**完整有序 trace**。
- DeepEval 把一切聚合成一次测试运行。

### 5.3 框架集成（12 个，同一模式）

模式都一样：**「插桩/替换客户端」+「在 evals_iterator 循环里调用 Agent」**，指标用 `TaskCompletionMetric`。各框架的插桩钩子：

| 框架 | 插桩方式 |
| --- | --- |
| 手动 | `@observe` + `update_current_trace(...)` |
| LangChain | `create_agent(...)`，invoke/ainvoke 传 `config={"callbacks":[CallbackHandler()]}` |
| LangGraph | `StateGraph`，invoke/ainvoke 传同一个 `CallbackHandler` |
| OpenAI | 把 `from openai import OpenAI` 换成 `from deepeval.openai import OpenAI`，调用包在 `with trace():` |
| Pydantic AI | `Agent(..., instrument=DeepEvalInstrumentationSettings())` |
| AgentCore | 先 `instrument_agentcore()`（也覆盖内部 Strands） |
| Strands | 先 `instrument_strands()` |
| Anthropic | 把 `from anthropic import Anthropic` 换成 `from deepeval.anthropic import Anthropic`，包 `with trace():` |
| LlamaIndex | `instrument_llama_index(instrument.get_dispatcher())`；`agent.run` 仅 async |
| OpenAI Agents | `add_trace_processor(DeepEvalTracingProcessor())` + deepeval 的 `Agent`/`function_tool` shim |
| Google ADK | 先 `instrument_google_adk()`；`runner.run_async` 仅 async |
| CrewAI | 先 `instrument_crewai()` + deepeval 的 `Crew`/`Agent`/`@tool` shim |

**代表例 — LangChain（Async）：**

```python
import asyncio
from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = create_agent(model="openai:gpt-4o-mini", tools=[multiply], system_prompt="Be concise.")

async def run_agent(prompt: str):
    return await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"callbacks": [CallbackHandler()]},
    )

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric()]):
    task = asyncio.create_task(run_agent(golden.input))
    dataset.evaluate(task)
```

**代表例 — OpenAI（Sync，drop-in 替换）：**

```python
from deepeval.openai import OpenAI
from deepeval.tracing import trace
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import TaskCompletionMetric

client = OpenAI()
for golden in dataset.evals_iterator(
    metrics=[TaskCompletionMetric()],
    async_config=AsyncConfig(run_async=False),
):
    with trace():
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": golden.input}],
        )
```

> Sync 统一靠 `async_config=AsyncConfig(run_async=False)`；仅 async 的框架（LlamaIndex、Google ADK）sync 版用 `asyncio.run(...)` 包一层。

### 5.4 组件级评测（Component-level）

- 轨迹评测告诉你「整条路径是否有效」；组件级评测用**同一条 trace** 逐个 span 打分，定位是哪个 planner / retriever / tool call / LLM generation / sub-agent 导致失败。
- Agent 被追踪后，**只给需要诊断的 span 挂组件指标**。轨迹分与组件分可在**同一次测试运行**里一起跑。

### 5.5 放进测试套件（pytest 写法）+ inspect

```python
# test_agent.py
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import Golden
from deepeval import assert_test
from agent import my_agent
import pytest

goldens = [Golden(input="Plan a three-day trip to Paris")]

@pytest.mark.parametrize("golden", goldens)
def test_agent(golden: Golden):
    my_agent(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

照样用 `deepeval test run` 运行。

> 💡 每次 `evals_iterator()` 运行都会快照到磁盘，可用裸命令 `deepeval inspect` 打开 **trace-tree TUI**，查看轨迹分、逐 span 分、指标理由。

---

## 6. 生产轨迹评测（Production）

上面的测试只覆盖「你想到的要测的」。应用上线后，同样的 trace 可送到 Confident AI，它在生产中对 **trace / span / thread（对话）** 跑评测，浮现失败的那些，使其成为你下一批测试用例。

```bash
CONFIDENT_API_KEY="confident_us..."
```

设好后，Agent 产生的每条 trace 都会发送到 Confident AI，失败 trace 会在那里显示。DeepEval 的 tracing 是**非侵入式**的，不影响你的代码。

三个评测层级：

- **Trace 级**：用端到端评测打「可观察结果」，或用轨迹评测分析「完整内部路径」。
- **Span 级（组件级）**：span 组成 trace，对 span 评测 = 评单个组件（如工具调用、检索）。
- **Thread 级（对话）**：thread 由一条或多条 trace 组成，表示一次多轮交互，评整段对话。

---

## 7. 下一步

- **核心概念**（构建可重复评测套件）：Test cases / Metrics / Datasets。
- **按用例的快速入门**：AI agents / Voice agents / Chatbots / RAG。
- **更多工作流**：生成合成数据（synthetic data）、模拟对话（simulate conversations）、框架集成。
- 已在生产：Confident AI 在还没人写评测前就浮现失败 trace，并转成下一轮数据集的 golden。

---

## 8. FAQ

- **评测卡住了？** 多数 LLM-as-a-judge 指标要调用评测模型。若供应商限流、超额、响应慢，评测看起来就像卡住。检查模型 key、配额、网络。
- **必须用 Confident AI 吗？** 不。DeepEval 本地运行；Confident AI 可选，生产阶段才用来浮现失败 trace 并转成下一轮评测。
- **测试文件放哪？** 任何测试运行器能发现的地方，通常紧挨应用或放 `tests/`。然后 `deepeval test run path/to/test_file.py`。
- **能用 OpenAI 以外的模型吗？** 能。支持多供应商及自定义/本地模型；OpenAI 只是许多示例最快的默认路径。
- **之后读什么？** 评 Agent → 先读 tracing；建可重复套件 → 先读 datasets 与 metrics。

---

## 9. 对本项目的落地建议

- 运行一律走 uv：`uv run deepeval test run ...`、`uv run python main.py`、`uv run deepeval inspect`。
- 你同时加了 `ragas`：那是另一个**偏 RAG** 的评测框架（faithfulness / answer relevancy / context precision 等），与 DeepEval 互补。DeepEval 更全（含 Agent 轨迹/组件、CI、生产），RAG 专项可两者对照。
- 建议目录：`src/ai_evaluation/` 放被评应用与插桩，`tests/` 放 `test_*.py`，`docs/evaluation-frameworks/` 放调研文档（本系列）。
- 最小起步：先跑通第 3 节的单轮 `GEval` 例（需 `OPENAI_API_KEY`），再上第 5 节的 Agent 轨迹例。