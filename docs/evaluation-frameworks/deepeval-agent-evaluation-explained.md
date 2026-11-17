# DeepEval Agent 评测讲解梳理(轨迹 + 组件)

> 对应材料:`deepeval-getting-started.md` 第 5 节「评测 AI Agent(轨迹 + 组件)」
> 官方原文:https://deepeval.com/docs/getting-started
> 姊妹文档:`deepeval-ai-agent-evaluation.md`(官方 Agent Evaluation 指南全译,三 scope + 六指标详解)
> 本文定位:讲解提纲与学习路线,不是 API 参考。API 细节查上面两份。

---

## 0. 动机:为什么需要这一节

单轮/多轮 case 的评估单位都是**文本**(一对 input/output,或一串 turns)。但 Agent 是「推理 → 调工具 → 观察 → 再推理」的循环,失败可能发生在**中间任何一步**,而最终答案看起来还行——例如工具报错后 agent 自我修复了,答案对了但烧掉 3 倍 token 和时间。

所以评估单位必须再次升级:

```
单轮:LLMTestCase            = 一张问答快照
多轮:ConversationalTestCase  = 一段对话记录
Agent:trace(轨迹)           = 一棵执行树(span tree)
```

这与 Langfuse 的 trace/observation 是**同一个概念**——DeepEval 的 `@observe` 就是在生成这种树。也是调研报告(README.md 4.4 节)里「agent 评估的单位是 trace 而非 completion」的落地。

---

## 1. 主线:五个环节的流水线

```
Golden(考题) → @observe 插桩(记录执行) → evals_iterator(跑题+收 trace)
→ 指标按 scope 打分(轨迹/组件) → 聚合结果(inspect 查看)
```

### 环节① Golden ——「只有题目的考卷」(getting-started 5.1)

```python
from deepeval.dataset import EvaluationDataset, Golden
dataset = EvaluationDataset(goldens=[Golden(input="Plan a three-day trip to Paris")])
```

`Golden` **只有 input,没有 actual_output**。这是与 `LLMTestCase` 的本质区别:

| | LLMTestCase | Golden |
|---|---|---|
| 内容 | 题目 + 答案都准备好 | 只有题目 |
| 指标做什么 | 只阅卷 | 答案由被测 agent 运行时真实产生,trace 顺手被捕获 |

这就是「离线基准测试」的标准形态:固定一批代表性任务(goldens),每次改 prompt/换模型后重跑,比较分数抓回归。

### 环节② 插桩 —— 没有 trace,轨迹指标就是瞎子(5.2)

两个动作:

```python
from deepeval.tracing import observe, update_current_trace

@observe()                      # 包住 agent 顶层函数(内部 LLM 调用/工具函数也包)
def my_ai_agent(query: str):
    answer = ...                # 真实 LLM 调用
    update_current_trace(input=query, output=answer)   # 给树根补上题目和最终答案
    return answer
```

- `@observe()` → 自动生成嵌套 span 树(agent span 包含 llm span、tool span);
- `update_current_trace(...)` → 设置轨迹级测试用例字段。

**关键因果**:TaskCompletionMetric 这类轨迹指标**必须**有 trace 才能工作——它要读完整有序的执行链,不是只看最终文本。这是初学者最容易跳过、然后报错的地方。

### 环节③ evals_iterator ——「跑题循环」(5.2)

```python
from deepeval.metrics import TaskCompletionMetric

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric()]):
    my_ai_agent(golden.input)   # 循环体内跑 agent,trace 自动捕获 + 打分
```

- 它把「遍历数据集 → 运行 agent → 捕获 trace → 跑指标 → 聚合」打包成一个循环;
- **Async vs Sync**:默认并发跑 goldens(快);`async_config=AsyncConfig(run_async=False)` 一次一个(调试/限流供应商用);
- **易混点 1:运行方式变了**。这是普通脚本,`uv run python main.py`,**不是** `deepeval test run`。

Sync 完整最小例:

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

### 环节④ 框架集成 —— 12 个框架,一个模式(5.3)

所有集成都是同一句话:**「用该框架的钩子替代手写 @observe,循环部分一字不改」**。钩子分三类:

| 类型 | 例子 |
|---|---|
| 回调/处理器 | LangChain/LangGraph:invoke 传 `config={"callbacks": [CallbackHandler()]}` |
| drop-in 替换客户端 | `from deepeval.openai import OpenAI` / `from deepeval.anthropic import Anthropic`,调用包在 `with trace():` |
| 全局 instrument 函数 | `instrument_strands()` / `instrument_crewai()` / `instrument_google_adk()` 等 |

讲解策略:只演示手动 + OpenAI 两个,其余查表即可——模式会了,表格只是速查。
(本项目 `test_real_llm.py` 用的原生 openai client,改一行 import 就能插桩。)

### 环节⑤ 组件级 —— 同一棵树,两种读数(5.4)

本节的思想高潮,对应 agent-evaluation 指南的三 scope:

| scope | 指标挂哪 | 读什么 | 回答什么 |
|---|---|---|---|
| 端到端(黑盒) | 普通指标 | 只看 input/output | 最终结果好不好 |
| **轨迹** | `evals_iterator(metrics=[...])` | **整棵 span 树** | 任务完成了吗、路径高效吗(TaskCompletion / StepEfficiency / PlanQuality / PlanAdherence) |
| **组件** | 具体 span:`@observe(type="llm", metrics=[...])` | **单个节点** | 哪一步选错工具/写错参数(ToolCorrectness / ArgumentCorrectness) |

```python
# 组件指标挂在做出工具调用决策的 LLM span 上
@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_llm(messages): ...
```

**同一次运行同时出两种分**:轨迹分告诉你「病了」,组件分告诉你「病灶在哪」。这与 RAG 拆 faithfulness/context 指标定位「检索端 vs 生成端」是完全同构的诊断逻辑。

---

## 2. 收尾:两种运行形态 + inspect(5.5)

**易混点 2:同一件事有两种写法**:

| | 脚本形态(5.2) | pytest 形态(5.5) |
|---|---|---|
| 循环 | `evals_iterator` | `@pytest.mark.parametrize` |
| 断言 | 循环内自动 | `assert_test(golden=golden, metrics=[...])` |
| 运行 | `uv run python main.py` | `uv run deepeval test run` |
| 适合 | 批量实验/探索 | CI 门禁(本项目已有 case 的形态) |

pytest 形态:

```python
# test_agent.py
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import TaskCompletionMetric
from agent import my_agent   # 已被 @observe 插桩

goldens = [Golden(input="Plan a three-day trip to Paris")]

@pytest.mark.parametrize("golden", goldens)
def test_agent(golden: Golden):
    my_agent(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

注意 `assert_test` 传的是 **golden** 而不是 test_case——trace 已被 `@observe` 捕获,DeepEval 自动组装测试用例。

调试工具:`deepeval inspect` 打开 **trace-tree TUI**,逐 span 看分数和 reason,是调试 agent 失败的第一工具(每次 evals_iterator 运行都会快照到磁盘)。

---

## 3. 与已有知识的连接

| 已学 | 本节新增 | 连接点 |
|---|---|---|
| `test_example.py` / `test_real_llm.py`(单轮) | Golden + trace | 评估单位从文本快照 → 执行树 |
| `test_conversation.py`(多轮) | thread 级评估 | 多轮对话 = 多条 trace 组成的 thread(getting-started 第 6 节) |
| Langfuse trace/observation | `@observe` span 树 | 同一概念,不同实现;分数可回写 Langfuse |
| RAG 三元组拆维度诊断 | 轨迹分 + 组件分 | 同构逻辑:拆维度是为了定位故障,不是凑总分 |
| 内置 verdict 型 vs GEval 型指标 | TaskCompletion 等轨迹指标 | 轨迹指标同样必须依赖 tracing,不能独立使用 |

---

## 4. 建议的讲解/动手顺序

1. **讲动机**(评估单位三级跳:文本 → 对话 → 执行树),5 分钟;
2. **跑通 Sync 手动插桩最小例**(避开 async 复杂度),用 `deepeval inspect` 看 span 树——「看见树」比讲十句都管用;
3. **加一个工具函数**(如 `search_flights`),故意让它出错,观察 TaskCompletion 分数变化 + 组件指标定位病灶;
4. **改写成 pytest 形态**,接进已有测试套件,形成 CI 门禁。

对应调研报告(README.md)学习路径的阶段 4 第 2 步。下一步可在 `src/deepeval_cases/` 落地一个可运行的最小 agent case(带一两个工具、真实 LLM 调用、轨迹 + 组件两种指标)。

---

## 5. 易错点清单

- 轨迹指标(TaskCompletion/StepEfficiency 等)**不能**脱离 tracing 独立使用,必须经 `evals_iterator` 或 `@observe` 配合;
- `evals_iterator` 脚本用 `python` 跑,pytest 形态才用 `deepeval test run`;
- 组件指标要挂在**做出该决策的 span** 上(工具选择/参数生成挂在 LLM span,不是 tool span);
- Sync 模式统一靠 `AsyncConfig(run_async=False)`;仅 async 的框架(LlamaIndex、Google ADK)sync 版用 `asyncio.run(...)` 包一层;
- `assert_test(golden=...)` 与 `assert_test(test_case, ...)` 是两种签名,取决于 trace 是否已自动捕获。