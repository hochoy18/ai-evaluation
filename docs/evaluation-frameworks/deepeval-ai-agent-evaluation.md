# AI Agent 评测（AI Agent Evaluation）

> 原文：Confident AI / DeepEval 官方文档 —《AI Agent Evaluation》（路径 `/guides/guides-ai-agent-evaluation`）
> 本文为详细中文翻译。代码示例保留原样，注释已汉化；原文中的示意图/视频以文字说明代替。

---

**AI Agent 评测**是衡量一个 Agent 在「推理、选择与调用工具、完成任务」上表现的过程——评测的范围与每一种失败模式相对应，从而让你能精确定位到底是哪里出了问题。但首先，什么是 AI Agent？

AI Agent 是一个由 LLM 驱动的系统，能够自主地对任务进行推理、制定计划，并借助外部工具执行动作，以达成用户目标。与只响应单次提示（prompt）的简单 LLM 应用不同，Agent 以**循环**方式运行——推理、行动、观察结果、调整策略，直到任务完成。

> ℹ️ **说明**：AI Agent 由两层构成：**推理层**（由 LLM 驱动）负责规划与决策；**行动层**（由 function calling 等工具驱动）在真实世界中执行动作。这两层协同迭代，直到任务完成。

由于 Agent 的成功同时取决于推理与行动，AI Agent 评测采用三种互补的**范围（scope）**：

- **端到端（黑盒输出）**：只评测 Agent 可观察的输入与最终输出，不检查内部执行过程。
- **基于轨迹（Trajectory-based）**：评测完整、有序的执行轨迹——包括推理、工具调用、观察结果与中间步骤——以判断 Agent 是「如何」得到结果的。
- **组件级（Component-level / 逐步）**：孤立地评测单个 span 或单个决策，例如「选择工具并生成其参数」的那个 LLM span。

本指南用**轨迹指标**评测完整轨迹的规划与执行质量，用**组件级指标**评测单个工具调用决策。完整的范围模型参见[基于轨迹的评测](/docs/evaluation-trajectory-based-llm-evals)。

无论你用的是什么技术栈，下文都提供了 **10+ 主流 Agent 框架**的真实、可直接复制的代码示例——LangChain、LangGraph、OpenAI Agents、Pydantic AI、CrewAI、Google ADK 等。

*关于每个 Agent 指标的全面解析，参见 [AI Agent 评测指标指南](/guides/guides-ai-agent-evaluation-metrics)。要理解这一切在 Agent 技术栈中的位置，请阅读[什么是评测框架（eval harness）、它为什么重要](/blog/what-is-an-eval-harness)。*

---

## AI Agent 流水线中的常见陷阱

一个 AI Agent 流水线包含「推理（规划）」与「行动（工具调用）」两类步骤，二者迭代直到任务完成。**推理层**决定「做什么」，**行动层**负责「怎么做」。

*（原文此处为 Agent 架构示意图：推理层 LLM 与行动层 Tools 循环交互。）*

**推理层**包含你的 LLM，负责理解任务、制定计划、决定使用哪些工具。**行动层**包含你的工具（function call、API、数据库等），负责执行这些决策。二者一起循环，直到任务完成或失败。

### 推理层（Reasoning Layer）

推理层由 LLM 驱动，负责规划与决策，通常涉及：

1. **理解用户意图**：分析输入，确定底层任务与目标。
2. **分解复杂任务**：拆成更小、可管理的子任务，可串行或并行执行。
3. **制定连贯策略**：勾勒出完成任务所需的步骤。
4. **决定使用哪些工具**：根据当前上下文决定工具及其使用顺序。

Agent 推理质量主要受以下因素影响：

- **LLM 选择**：不同模型推理能力不同。像 `gpt-4o`、`claude-3.5-sonnet` 这样的大模型通常比小模型推理更好，但成本与延迟更高。
- **提示模板（Prompt template）**：给 LLM 的系统提示与指令极大地影响它处理任务的方式。精心设计的提示能引导 LLM 逐步推理、考虑边界情况、产出连贯的计划。
- **温度（Temperature）**：低温度产生更确定、更聚焦的推理；高温度可能更有创意，但计划可能不一致。

> 💡 **提示**：在改进推理层时，提示模板可以说是最重要的因素。

以下是 AI Agent 评测在推理层要回答的关键问题：

- **Agent 是否制定了有效的计划？** 好计划应当逻辑清晰、完整、高效。糟糕的计划会导致浪费步骤、遗漏需求，甚至直接失败。
- **计划的粒度是否合适？** 太细的计划浪费资源，太粗的计划会遗漏关键细节。
- **计划是否考虑了依赖关系？** 某些子任务必须先于其他任务完成，好计划会尊重这些依赖。
- **Agent 是否遵循了自己的计划？** 制定了好计划却在执行中偏离它，等于自我拆台。

### 行动层（Action Layer）

行动层是 Agent 通过工具（function call、API、数据库等）与外部系统交互的地方，也是最常出问题的地方。行动层通常涉及：

1. **选对工具**：根据当前子任务，从可用选项中选出合适的工具。
2. **生成正确参数**：基于输入与上下文，为工具调用生成正确的参数。
3. **按正确顺序调用工具**：当操作之间存在依赖时。
4. **处理工具输出**：把结果回传给推理层。

Agent 工具调用质量主要受以下因素影响：

- **可用工具**：暴露给 Agent 的工具集合决定了它能采取哪些动作。工具太多会让 LLM 困惑，太少则可能留下能力缺口。
- **工具描述**：清晰、无歧义的描述帮助 LLM 理解何时、如何使用每个工具。含糊的描述会导致选错工具。
- **工具 Schema**：定义良好的输入/输出 schema（正确的类型、必填字段、示例）帮助 LLM 生成正确参数。
- **工具命名**：直观、描述性的名字（如 `SearchFlights` 对比 `api_call_1`）让 LLM 更容易选对工具。

> ⚠️ **注意**：工具使用失败是 AI Agent 最常见的问题之一。即使是最先进的 LLM，也可能在「选对工具、生成合法参数、遵守调用顺序」上出错。

以下是 AI Agent 评测在行动层要回答的关键问题：

- **Agent 是否选对了工具？** 有多个工具时，必须为每个子任务选最合适的那个。需要 `WebSearch` 时却选了 `Calculator`，会导致任务失败。
- **Agent 调用的工具数量是否合适？** 太少任务完不成；调用不必要的工具浪费资源、还可能引入错误。
- **Agent 是否按正确顺序调用工具？** 有些任务要求特定顺序——不能先订票再搜索可用航班。
- **Agent 是否提供了正确参数？** 即使选对了工具，参数错误也会失败。例如工具期望 `{"location": "San Francisco, CA, USA"}`，却传入 `{"city": "San Francisco"}`，可能报错或返回错误数据。
- **参数值是否从上下文中正确提取？** Agent 必须准确解析用户输入与先前的工具输出，构造合法参数。
- **工具描述是否足够清晰？** 含糊或不完整的描述会让 LLM 困惑于何时、如何使用工具。

### 整体执行（Overall Execution）

整体执行涵盖「推理层与行动层协同迭代」的 Agent 循环，涉及：

1. **编排推理—行动循环**：LLM 推理、调用工具、观察结果、再推理。
2. **优雅地处理错误与边界情况**：事情不如预期时调整策略。
3. **迭代直到任务完成**，或判定无法完成。

以下是 AI Agent 评测在整体执行上能回答的问题：

- **Agent 完成任务了吗？** 这是成功的终极衡量——是否达成了用户的要求。
- **Agent 执行是否高效？** 应无多余或冗余步骤。用相同参数反复调用同一工具、或绕远路完成简单目标，都是浪费时间与资源。
- **Agent 是否恰当处理失败？** 工具调用失败或返回意外结果时，应调整策略，而不是反复尝试同一套失败做法。
- **Agent 是否紧扣任务？** 应聚焦用户原始请求，而不是跑题或执行未被要求的动作。

---

## 开发阶段的 Agent 评测

开发阶段评测的核心是「用数据集 + 指标做基准测试（benchmarking）」。指标可以对黑盒输出、完整轨迹或单个组件打分；数据集则确保你在[同一组 golden（标准样例）](/docs/evaluation-datasets)上比较 Agent 的不同迭代。

开发阶段评测帮助回答这类问题：

- **哪个 Agent 版本表现最好？** 在同一数据集上并排比较不同实现。
- **改动提示会影响整体成功率吗？** 测试提示变体，衡量其对任务完成率的影响。
- **新工具是帮忙还是帮倒忙？** 评估新增/修改工具是否提升了 Agent 表现。
- **Agent 到底在哪里失败？** 精确定位问题源于规划不佳、选错工具，还是参数错误。

但首先，你得告诉 `deepeval` 你的 AI Agent 内部有哪些组件，指标才能运作。这通过 [LLM 追踪（tracing）](/docs/evaluation-llm-tracing)实现。LLM 追踪能帮 `deepeval` 勾勒出 Agent 的完整执行轨迹：只需给 Agent 内的函数加上 `@observe` 装饰器，且**不会给 Agent 增加任何延迟**。

*（原文此处为组件级评测示意图。）*

下面看一个例子：为一个使用 OpenAI 作为 LLM 的「机票预订 Agent」设置追踪：

```python
import json

from openai import OpenAI
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset

client = OpenAI()
tools = [...]  # 工具 schema 见下文

@observe(type="tool")
def search_flights(origin, destination, date):
    # 模拟航班搜索
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    # 模拟订票
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="llm")
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    return response

@observe(type="agent")
def travel_agent(user_input):
    messages = [{"role": "user", "content": user_input}]

    # LLM 推理决定调用哪个工具
    response = call_openai(messages)
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # 执行工具
    flights = search_flights(args["origin"], args["destination"], args["date"])

    # LLM 决定预订最便宜的航班
    cheapest = min(flights, key=lambda x: x["price"])
    messages.append({"role": "assistant", "content": f"Found flights. Booking cheapest: {cheapest['id']}"})

    booking = book_flight(cheapest["id"])

    return f"Booked flight {cheapest['id']} for ${cheapest['price']}. Confirmation: {booking['confirmation']}"
```

<details>
<summary>查看 OpenAI 工具 schema</summary>

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a specific flight by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"}
                },
                "required": ["flight_id"]
            }
        }
    }
]
```
</details>

在这个例子中，我们给 Agent 的每个组件都加了 `@observe()`，以创建完整执行轨迹：

- `search_flights` 与 `book_flight` 上的 `@observe(type="tool")`——标记为**工具 span**，代表 Agent 与外部系统交互的行动层。
- `call_openai` 上的 `@observe(type="llm")`——标记为 **LLM span**，捕获 OpenAI 决定调用哪个工具的推理层。
- `travel_agent` 上的 `@observe(type="agent")`——标记为编排整个流程的**顶层 agent span**。

调用 `travel_agent()` 时，`deepeval` 自动捕获嵌套执行：agent span 包含 LLM span（推理）与工具 span（行动），形成一棵可供指标分析的树状结构。

> 💡 **提示**：`type` 参数可选但推荐填写——它帮助 `deepeval` 理解你的 Agent 架构，并决定每个指标作用于哪些 span。不指定 type 时，默认为 custom span。

如果你设置了 `CONFIDENT_API_KEY` 或运行过 `deepeval login`，那么每次运行评测时，测试运行结果也会出现在 Confident AI 平台上。

*（原文此处为视频：在本地运行 Agent 评测，并在平台上获得测试报告。）*

### 评测推理层

`deepeval` 提供两个**轨迹指标**来评测 Agent 的推理与规划能力：

- [`PlanQualityMetric`](/docs/metrics-plan-quality)：评测 Agent 生成的**计划**是否逻辑清晰、完整、高效，能否完成给定任务。
- [`PlanAdherenceMetric`](/docs/metrics-plan-adherence)：评测 Agent 在执行中是否**遵循了自己的计划**，还是偏离了既定策略。

**需要组合使用这两个指标**，因为你要确保 Agent 既能制定好计划，又能一致地遵循它。评测推理层能确保 Agent 在行动开始前就有坚实基础。先在 `deepeval` 中创建这两个指标：

```python
from deepeval.metrics import PlanQualityMetric, PlanAdherenceMetric

plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()
```

> ℹ️ **说明**：`deepeval` 的所有指标都允许设置及格 `threshold`（阈值）、开启 `strict_mode` 与 `include_reason`，并可使用**任意** LLM 做评测。每个指标的算法细节见其各自文档页：
> - [`PlanQualityMetric`](/docs/metrics-plan-quality)
> - [`PlanAdherenceMetric`](/docs/metrics-plan-adherence)

最后，让你已追踪的 AI Agent 在你准备好的[数据集](/docs/evaluation-datasets)上循环，并把 `PlanAdherenceMetric` 与 `PlanQualityMetric` 定义为轨迹指标。它们需要追踪，因为要跨完整有序轨迹检查规划与执行：

```python
from deepeval.dataset import EvaluationDataset, Golden

# 创建数据集
dataset = EvaluationDataset(goldens=[
    Golden(input="Book a flight from NYC to London for next Monday")
])

# 带指标循环数据集
for golden in dataset.evals_iterator(metrics=[plan_quality, plan_adherence]):
    travel_agent(golden.input)
```

本例中的 `travel_agent` 可以是任何被 `@observe` 装饰的 Agent。无论 `evals_iterator` 内运行的是哪个被装饰的函数，`deepeval` 都会自动收集轨迹并在其上运行指定指标。

**恭喜 🎉！** 你已学会评测 AI Agent 的推理能力，接下来进入行动层。

### 评测行动层

`deepeval` 提供两个 LLM 评测指标来评测 Agent 的工具调用能力：

- [`ToolCorrectnessMetric`](/docs/metrics-tool-correctness)：基于「期望工具列表」，评测 Agent 是否**选对了工具**、并以期望的方式调用。
- [`ArgumentCorrectnessMetric`](/docs/metrics-argument-correctness)：基于输入与上下文，评测 Agent 是否为每次工具调用**生成了正确参数**。

这两个是**组件级指标**，应严格放在 Agent 的 **LLM 组件**上（如 `call_openai`），因为工具调用决策正是在这里做出的。LLM 负责选工具、生成参数——所以这正是我们要评测的地方。

> 📝 **注**：工具选择与参数生成同样关键——「选对工具但参数错」和「 outright 选错工具」一样有问题。

先定义指标：

```python
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()
```

然后，把指标加到 AI Agent 的 **LLM 组件**上：

```python
# 给 @observe 加上 metrics=[...]
@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    return response
```

最后，运行已加指标的、被追踪的 AI Agent：

```python
from deepeval.dataset import EvaluationDataset, Golden

# 创建数据集
dataset = EvaluationDataset(goldens=[
    Golden(input="What's the weather like in San Francisco and should I bring an umbrella?")
])

# 用行动层指标评测
for golden in dataset.evals_iterator():
    weather_agent(golden.input)
```

> 译注：原文此处循环里写的是 `weather_agent`，与上文定义的 `travel_agent` 不一致，应为示例笔误；实际指「被 `@observe` 装饰、且 LLM 组件挂了行动层指标」的那个 Agent。

`tools_called` 包含 Agent 实际调用的工具（及其参数），`expected_tools` 定义「本应调用哪些工具」。计算方式见各自文档页：

- [`ToolCorrectnessMetric`](/docs/metrics-tool-correctness)
- [`ArgumentCorrectnessMetric`](/docs/metrics-argument-correctness)

接下来评测整体执行。

> ⚠️ **注意**：使用 `ToolCorrectnessMetric` 时，可通过 `evaluation_params` 配置严格程度。默认只比较工具名，但也可以要求输入参数与输出都匹配。

### 评测整体执行

`deepeval` 提供两个**轨迹指标**来评测 Agent 的整体执行：

- [`TaskCompletionMetric`](/docs/metrics-task-completion)：通过分析完整有序的执行轨迹，评测 Agent 是否**成功完成了目标任务**。
- [`StepEfficiencyMetric`](/docs/metrics-step-efficiency)：通过检查完整有序轨迹中是否有不必要或冗余步骤，评测 Agent 是否**高效地完成任务**。

> 📝 **注**：Agent 可能完成了任务但效率低下，浪费 token 与时间；反之，高效却没完成任务的 Agent 毫无价值。两个指标对全面评测执行都必不可少。

这两个指标分析完整有序的 Agent 轨迹以评估执行质量：

```python
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric

task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()
```

最后，和上面一样，带这些指标运行你的 AI Agent：

```python
from deepeval.dataset import EvaluationDataset, Golden

# 创建数据集
dataset = EvaluationDataset(goldens=[
    Golden(input="Book the cheapest flight from NYC to LA for tomorrow")
])

# 用执行指标评测完整轨迹
for golden in dataset.evals_iterator(metrics=[task_completion, step_efficiency]):
    travel_agent(golden.input)
```

`TaskCompletionMetric` 会评估 Agent 是否真的按要求订了机票，`StepEfficiencyMetric` 则评估它是否走了最直接的完成路径。

> ℹ️ **说明**：`TaskCompletionMetric` 与 `StepEfficiencyMetric` 都是轨迹指标，不能独立使用，**必须**通过 `evals_iterator` 或 `observe` 装饰器配合追踪使用。

---

## 端到端 vs 轨迹 vs 组件级评测

AI Agent 评测可在三种不同范围上运作：

- **端到端评测**：仅凭 Agent 的输入与输出对最终黑盒输出打分，不需要也不检查内部轨迹。
- **轨迹评测**：推理指标（`PlanQualityMetric`、`PlanAdherenceMetric`）与执行指标（`TaskCompletionMetric`、`StepEfficiencyMetric`）传给 `evals_iterator(metrics=[...])`。它们需要追踪，评测「推理、工具调用、观察结果及其他步骤」的完整有序序列。
- **组件级评测**：行动层指标（`ToolCorrectnessMetric`、`ArgumentCorrectnessMetric`）通过 `@observe(metrics=[...])` 直接挂在 LLM 组件的 `@observe` 装饰器上，评测「做出工具调用决策」的那个单一 span。

这种区分很重要，因为不同指标需要不同范围：

| 指标类型 | 范围 | 原因 |
| --- | --- | --- |
| 最终输出质量 | 端到端（黑盒） | 不依赖内部实现细节即可对结果打分 |
| 推理与执行 | 轨迹 | 需要完整有序轨迹来评估规划、遵循度、完成度与效率 |
| 工具选择与参数 | 组件级 | 对「发生工具调用决策」的单个 LLM span 打分 |

何时用哪种方式，详见[端到端评测](/docs/evaluation-end-to-end-llm-evals)、[基于轨迹的评测](/docs/evaluation-trajectory-based-llm-evals)、[组件级评测](/docs/evaluation-component-level-llm-evals)文档。

### 使用自定义评测

上面这些 Agent 指标很有用，但偏通用。如果你需要评测特定于自身用例的东西——比如 Agent 是否保持专业语气、是否遵守公司规范、是否清晰解释其推理——怎么办？

这就要用到 [`GEval`](/docs/metrics-llm-evals)。G-Eval 是一个「LLM-as-a-judge（用 LLM 当裁判）」框架，基于你用** plain English（自然语言）定义的任意自定义标准**来评测输出。根据测试用例与评测目标，它既可作用于黑盒输出，也可作用于特定组件。

---

## 在 CI/CD 中评测 AI Agent

Agent 在开发阶段通过后，你会希望在每次 push 或 PR 时防止回归。`deepeval` 通过 `assert_test()` 与 `deepeval test run` 命令接入 `pytest`，让你的 Agent 评测像普通测试套件一样运行——本地开发时如此，CI/CD 中自动运行也如此。

根据你的技术栈对 Agent 插桩，然后用 golden 对 `pytest` 测试做参数化并调用 `assert_test()`。由于 Agent 已被追踪，`deepeval` 会自动构建每个测试用例——无需手动提取输入输出。

### Python 各框架示例

#### 手动插桩（Manual Instrumentation）

```python
# test_llm_app.py
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe, update_current_trace

# 1. 加载 golden 数据集
dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

# 2. 给你的 Agent 插桩
@observe()
def my_ai_agent(query: str) -> str:
    answer = "Pi rounded to 2 decimal places is 3.14."
    update_current_trace(input=query, output=answer)
    return answer

# 3. 在每个 golden 上做端到端评测
@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden: Golden):
    my_ai_agent(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

用 `@observe` 包裹 Agent 的顶层函数，并用 `update_current_trace(...)` 设置轨迹级测试用例字段。完整接口见 [LLM 追踪](/docs/evaluation-llm-tracing)。

#### LangChain

```python
# test_langchain_app.py
import pytest
from langchain.agents import create_agent
from deepeval import assert_test
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt="Answer math questions concisely.",
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_langchain_app(golden: Golden):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

把 `deepeval` 的 `CallbackHandler` 传给 Agent `invoke` 方法。完整接口见 [LangChain 集成](/integrations/frameworks/langchain)。

#### LangGraph

```python
# test_langgraph_app.py
import pytest
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from deepeval import assert_test
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

llm = init_chat_model("openai:gpt-4o-mini")

def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = (
    StateGraph(MessagesState)
    .add_node(chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_langgraph_app(golden: Golden):
    graph.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

把 `deepeval` 的 `CallbackHandler` 传给 `StateGraph` 的 `invoke` 方法。完整接口见 [LangGraph 集成](/integrations/frameworks/langgraph)。

#### OpenAI

```python
# test_openai_app.py
import pytest
from deepeval import assert_test
from deepeval.openai import OpenAI
from deepeval.tracing import trace
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

# 直接替换：把 from openai import OpenAI 换成下面这行
client = OpenAI()

@pytest.mark.parametrize("golden", dataset.goldens)
def test_openai_app(golden: Golden):
    with trace():
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": golden.input},
            ],
        )
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

把 `from openai import OpenAI` 直接替换为 `from deepeval.openai import OpenAI`——每次 completion 调用自动成为一个 LLM span。完整接口见 [OpenAI 集成](/integrations/frameworks/openai)。

#### Pydantic AI

```python
# test_pydantic_ai_app.py
import pytest
from pydantic_ai import Agent
from deepeval import assert_test
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

agent = Agent(
    "openai:gpt-5",
    system_prompt="Answer in one short sentence.",
    instrument=DeepEvalInstrumentationSettings(),
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_pydantic_ai_app(golden: Golden):
    agent.run_sync(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

给 `Agent` 的 `instrument` 关键字传入 `DeepEvalInstrumentationSettings()`。完整接口见 [Pydantic AI 集成](/integrations/frameworks/pydanticai)。

#### AgentCore

```python
# test_agentcore_app.py
import pytest
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from deepeval import assert_test
from deepeval.integrations.agentcore import instrument_agentcore
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

instrument_agentcore()

app = BedrockAgentCoreApp()
agent = Agent(model="amazon.nova-lite-v1:0")

@app.entrypoint
def invoke(payload):
    result = agent(payload["prompt"])
    return {"result": result.message}

@pytest.mark.parametrize("golden", dataset.goldens)
def test_agentcore_app(golden: Golden):
    invoke({"prompt": golden.input})
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

在创建 AgentCore app 之前调用 `instrument_agentcore()`——它也会为运行在 AgentCore 内的 [Strands](https://strandsagents.com/) Agent 插桩。完整接口见 [AgentCore 集成](/integrations/frameworks/agentcore)。

#### Strands

```python
# test_strands_agent.py
import pytest
from strands import Agent
from strands.models.openai import OpenAIModel
from deepeval import assert_test
from deepeval.integrations.strands import instrument_strands
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="Help me return my order.")])

instrument_strands()

agent = Agent(
    model=OpenAIModel(model_id="gpt-4o-mini"),
    system_prompt="You are a helpful assistant.",
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_strands_agent(golden: Golden):
    agent(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

在创建或调用 Agent 之前调用 `instrument_strands()`（AgentCore 托管的 Strands 请用 AgentCore 那一节）。完整接口见 [Strands 集成](/integrations/frameworks/strands)。

#### Anthropic

```python
# test_anthropic_app.py
import pytest
from deepeval import assert_test
from deepeval.anthropic import Anthropic
from deepeval.tracing import trace
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

# 直接替换：把 from anthropic import Anthropic 换成下面这行
client = Anthropic()

@pytest.mark.parametrize("golden", dataset.goldens)
def test_anthropic_app(golden: Golden):
    with trace():
        client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system="Answer in one short sentence.",
            messages=[{"role": "user", "content": golden.input}],
        )
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

把 `from anthropic import Anthropic` 直接替换为 `from deepeval.anthropic import Anthropic`——每次 `messages.create(...)` 调用自动成为一个 LLM span。完整接口见 [Anthropic 集成](/integrations/frameworks/anthropic)。

#### LlamaIndex

```python
# test_llamaindex_app.py
import asyncio
import pytest
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
import llama_index.core.instrumentation as instrument
from deepeval import assert_test
from deepeval.integrations.llama_index import instrument_llama_index
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

instrument_llama_index(instrument.get_dispatcher())

agent = FunctionAgent(
    tools=[],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="Answer math questions concisely.",
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_llamaindex_app(golden: Golden):
    asyncio.run(agent.run(golden.input))
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

把 `deepeval` 的事件处理器注册到 LlamaIndex 的 instrumentation dispatcher 上。完整接口见 [LlamaIndex 集成](/integrations/frameworks/llamaindex)。

#### OpenAI Agents

```python
# test_openai_agents_app.py
import pytest
from agents import Runner, add_trace_processor
from deepeval import assert_test
from deepeval.openai_agents import Agent, DeepEvalTracingProcessor
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

add_trace_processor(DeepEvalTracingProcessor())

agent = Agent(
    name="math_agent",
    instructions="Answer math questions concisely.",
)

@pytest.mark.parametrize("golden", dataset.goldens)
def test_openai_agents_app(golden: Golden):
    Runner.run_sync(agent, golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

注册一次 `DeepEvalTracingProcessor`，然后用 `deepeval` 的 `Agent` shim 构建 Agent。完整接口见 [OpenAI Agents 集成](/integrations/frameworks/openai-agents)。

#### Google ADK

```python
# test_google_adk_app.py
import asyncio
import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from deepeval import assert_test
from deepeval.integrations.google_adk import instrument_google_adk
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

instrument_google_adk()

agent = LlmAgent(model="gemini-2.0-flash", name="assistant", instruction="Answer math questions concisely.")
runner = InMemoryRunner(agent=agent, app_name="deepeval-google-adk")

async def run_agent(prompt: str) -> str:
    session = await runner.session_service.create_session(app_name="deepeval-google-adk", user_id="demo-user")
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo-user", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content:
            return "".join(part.text for part in event.content.parts if getattr(part, "text", None))
    return ""

@pytest.mark.parametrize("golden", dataset.goldens)
def test_google_adk_app(golden: Golden):
    asyncio.run(run_agent(golden.input))
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

在构建 `LlmAgent` 之前调用一次 `instrument_google_adk()`。完整接口见 [Google ADK 集成](/integrations/frameworks/google-adk)。

#### CrewAI

```python
# test_crewai_app.py
import pytest
from crewai import Task
from deepeval import assert_test
from deepeval.integrations.crewai import instrument_crewai, Crew, Agent
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

dataset = EvaluationDataset(goldens=[Golden(input="What is pi rounded to 2 decimal places?")])

instrument_crewai()

tutor = Agent(
    role="Math Tutor",
    goal="Answer math questions accurately and concisely.",
    backstory="An experienced tutor who explains simple math clearly.",
)
task = Task(
    description="{question}",
    expected_output="Pi rounded to 2 decimal places is 3.14.",
    agent=tutor,
)
crew = Crew(agents=[tutor], tasks=[task])

@pytest.mark.parametrize("golden", dataset.goldens)
def test_crewai_app(golden: Golden):
    crew.kickoff({"question": golden.input})
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
```

调用一次 `instrument_crewai()`，然后用 `deepeval` 的 `Crew` 与 `Agent` shim 构建 crew。完整接口见 [CrewAI 集成](/integrations/frameworks/crewai)。

### TypeScript 各框架示例

下面每个 tab 都用同一个 matcher：golden 是被评测对象，`task` 产生「被拿来对照评判」的轨迹。导入 `deepeval/vitest` 会注册 `toPass()`，而 `npx deepeval test run` 会替你注入它。

#### 手动插桩（Manual Instrumentation）

```typescript
// llm-app.test.ts
import { it, expect } from "vitest";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import { observe, updateCurrentTrace } from "deepeval/tracing";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const myAiAgent = observe({
  type: "agent",
  fn: async (query: string): Promise<string> => {
    const answer = "Pi rounded to 2 decimal places is 3.14.";
    updateCurrentTrace({ input: query, output: answer });
    return answer;
  },
});

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => myAiAgent(g.input),
    });
  },
);
```

用 `observe` 包裹 Agent 顶层函数，并用 `updateCurrentTrace(...)` 设置轨迹级测试用例字段。完整接口见 [LLM 追踪](/docs/evaluation-llm-tracing)。

#### LangChain

```typescript
// langchain-app.test.ts
import { it, expect } from "vitest";
import { createAgent } from "langchain";
import { DeepEvalCallbackHandler } from "deepeval/integrations/langchain";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const agent = createAgent({
  model: "openai:gpt-4o-mini",
  tools: [],
  systemPrompt: "Answer math questions concisely.",
});

const handler = new DeepEvalCallbackHandler({});

const ask = (prompt: string) =>
  agent.invoke(
    { messages: [{ role: "user", content: prompt }] },
    { callbacks: [handler] },
  );

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => ask(g.input),
    });
  },
);
```

把 `DeepEvalCallbackHandler` 传给 Agent 的 `invoke`。完整接口见 [LangChain 集成](/integrations/frameworks/langchain)。

#### Mastra

```typescript
// mastra-app.test.ts
import { it, expect } from "vitest";
import { Observability } from "@mastra/observability";
import { Mastra } from "@mastra/core/mastra";
import { DeepEvalExporter } from "deepeval/integrations/mastra";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const mastra = new Mastra({
  agents: { mathAgent },
  observability: new Observability({
    configs: {
      deepeval: {
        serviceName: "math-app",
        exporters: [new DeepEvalExporter()],
      },
    },
  }),
});

const ask = (prompt: string) => mastra.getAgent("mathAgent").generate(prompt);

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => ask(g.input),
    });
  },
);
```

`toPass()` 会在打分前等待 exporter 落定，因此测试内无需手动 flush。完整接口见 [Mastra 集成](/integrations/frameworks/mastra)。

#### LangGraph

```typescript
// langgraph-app.test.ts
import { it, expect } from "vitest";
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { DeepEvalCallbackHandler } from "deepeval/integrations/langchain";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const graph = new StateGraph(MessagesAnnotation)
  .addNode("chatbot", chatbot)
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END)
  .compile();

const handler = new DeepEvalCallbackHandler({});

const ask = (prompt: string) =>
  graph.invoke(
    { messages: [{ role: "user", content: prompt }] },
    { callbacks: [handler] },
  );

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => ask(g.input),
    });
  },
);
```

同一个 `DeepEvalCallbackHandler` 也覆盖 LangGraph，因为 LangGraph 跑在 LangChain 的 callback 系统上。完整接口见 [LangGraph 集成](/integrations/frameworks/langgraph)。

#### OpenAI

```typescript
// openai-app.test.ts
import { it, expect } from "vitest";
import { OpenAI } from "openai";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { AnswerRelevancyMetric } from "deepeval/metrics";
import { instrumentOpenAI } from "deepeval/openai";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const client = new OpenAI();
instrumentOpenAI(client);

const respond = (prompt: string) =>
  client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
  });

it.each(dataset.goldens as Golden[])(
  "answers the question #%$",
  async (golden) => {
    await expect(golden).toPass([new AnswerRelevancyMetric()], {
      task: (g) => respond(g.input),
    });
  },
);
```

`instrumentOpenAI` 原地 patch client，使其每次调用都成为一个 LLM span。完整接口见 [OpenAI 集成](/integrations/frameworks/openai)。

#### OpenAI Agents

```typescript
// openai-agents-app.test.ts
import { it, expect } from "vitest";
import { Agent, run, addTraceProcessor } from "@openai/agents";
import { DeepEvalTracingProcessor } from "deepeval/integrations/openai-agents";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

addTraceProcessor(new DeepEvalTracingProcessor());

const agent = new Agent({
  name: "mathAgent",
  instructions: "Answer math questions concisely.",
  model: "gpt-4o-mini",
});

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => run(agent, g.input),
    });
  },
);
```

注册一次 processor，每次 `run(...)` 都成为一条轨迹。完整接口见 [OpenAI Agents 集成](/integrations/frameworks/openai-agents)。

#### Vercel AI SDK

```typescript
// ai-sdk-app.test.ts
import { it, expect } from "vitest";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { OpenTelemetry } from "@ai-sdk/otel";
import { configureAiSdkTracing } from "deepeval/integrations/ai-sdk";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What is pi rounded to 2 decimal places?" })],
});

const telemetry = {
  integrations: [
    new OpenTelemetry({ tracer: configureAiSdkTracing({ name: "math-app" }) }),
  ],
};

const ask = (prompt: string) =>
  generateText({
    model: openai("gpt-4o-mini"),
    prompt,
    telemetry,
  });

it.each(dataset.goldens as Golden[])(
  "completes the task #%$",
  async (golden) => {
    await expect(golden).toPass([new TaskCompletionMetric()], {
      task: (g) => ask(g.input),
    });
  },
);
```

Telemetry 按调用 opt-in：没有 `isEnabled` 与 `tracer` 的调用绝不会被追踪。完整接口见 [Vercel AI SDK 集成](/integrations/frameworks/ai-sdk)。

### 运行测试套件

然后用 `deepeval test run` 运行测试套件，它在 `pytest` 之上叠加了 LLM 专属特性：

```bash
deepeval test run test_llm_app.py
```

把这个命令放进 `.yml` 工作流，指标不及格就会让构建失败，从而在 Agent 回归上线前就抓住它。完整 CI/CD 演练（含可直接用的 GitHub Actions 配置）见 [AI Agent 评测快速入门](/docs/getting-started-agents)。

---

## 用编码助手评测 AI Agent

越来越多的 Agent 是由 Claude Code 这类编码助手「构建」出来的。但问题在于：编码 Agent 的 harness——`CLAUDE.md`、hooks、skills、plugins、MCP servers——是为了「把正确的上下文喂给模型、产出好代码」而设计的。它**天生没有「拿 Agent 输出对照 ground truth 衡量」的概念**。测试非绿即红，但没有一个 `assertEqual` 能判断「这条回复是否忠实？」

*（原文此处为 Claude Code 终端演示。）*

这个缺口正是**评测框架（eval harness）**要填补的，而 DeepEval 就是 AI Agent 的 eval harness。由于 skill 按需加载，把它塞进你的编码 Agent 只需一行：

```bash
npx skills add confident-ai/deepeval --skill "deepeval"
```

此后，Claude Code 就能自己驱动整个循环——生成数据集、运行 `deepeval test run`、检查本地捕获的轨迹——把「加评测并修复失败项」变成它真正能执行的事。

*关于 eval harness 在 Agent 技术栈中的位置、以及它如何接入编码 Agent 自身 harness 的全景，请阅读[Eval harness：是什么、怎么用、为什么要在意](/blog/what-is-an-eval-harness)。*

---

## 生产阶段的 Agent 评测

在生产中，目标从「基准测试」转向「抓住那些失败的轨迹」。你的数据集只覆盖了你想到的失败，真实用户会找到其他失败。生产评测还需要：

- **异步运行**——绝不阻塞 Agent 的响应。
- **避免资源开销**——不在本地初始化指标、不在本地发起 LLM 裁判调用。
- **喂养下一轮评测**——每条失败轨迹都是你的数据集缺失的测试用例。

[Confident AI](https://www.confident-ai.com/products/llm-observability) 能在「还没人为这些失败写评测」之前就浮现它们，把它们聚类成失败模式，并转化为下一轮 `deepeval` 运行的 golden。

每当 Agent 运行，`deepeval` 都在后台把轨迹导出到 Confident AI。失败的轨迹正是值得拉进数据集的那些，于是下一轮开发评测就会测试「在生产中坏掉的那条轨迹」。

*（原文此处为视频：在 Confident AI 上发现失败的 Agent 轨迹，并将其转化为下一轮评测。）*

> 💡 **提示**：上手请在终端运行 `deepeval login`，并跟随 [Confident AI LLM 追踪设置指南](https://www.confident-ai.com/docs/llm-tracing/quickstart)。

---

## 结论

本指南中，你了解到 AI Agent 可能在多个层面失败：

- **推理层**——规划不佳、忽略依赖、偏离计划。
- **行动层**——选错工具、参数错误、调用顺序错误。
- **整体执行**——任务未完成、步骤低效、跑题。

为抓住这些问题，`deepeval` 提供了可在不同范围应用的指标：

| 范围 | 用途 | 示例指标 |
| --- | --- | --- |
| 端到端（黑盒） | 只评测最终输出 | 聚焦输出的指标与自定义 `GEval` 标准 |
| 轨迹 | 评测完整有序轨迹 | `PlanQualityMetric`、`PlanAdherenceMetric`、`TaskCompletionMetric`、`StepEfficiencyMetric` |
| 组件级 | 评测单个 span 或决策 | `ToolCorrectnessMetric`、`ArgumentCorrectnessMetric` |

> ℹ️ **开发 vs 生产**
> - **开发**——用数据集 + 本地定义的指标，对 Agent 迭代做基准测试与比较。
> - **生产**——把轨迹导出到 Confident AI，抓住失败的那些，并把它们转化为下一轮开发评测的 golden。

有了恰当的评测，你就能在用户之前抓住回归、精确定位 Agent 在哪里失败、用数据驱动决定上线哪个版本，并在生产中持续监控质量。

---

## 常见问题（FAQ）

**Q：什么是 AI Agent 评测？**
A：AI Agent 评测衡量一个自主 LLM 系统在「推理、规划、选择与调用工具、完成任务」上的表现。它可以对最终黑盒输出做端到端打分，把完整有序执行轨迹作为轨迹打分，或孤立评测单个组件 span（如 LLM 的工具调用决策）。

**Q：AI Agent 评测与普通 LLM 评测有何不同？**
A：标准 LLM 评测通常只对「一个输入-输出对」打分。AI Agent 评测还能对「推理步骤、工具调用、观察结果、中间决策」的完整有序轨迹打分，或孤立单个组件——从而精确定位失败究竟来自规划不佳、选错工具、参数错误，还是任务执行不完整。

**Q：在 DeepEval 中我该用哪些 AI Agent 指标？**
A：对大多数 Agent，先从推理层的 `PlanQualityMetric` 与 `PlanAdherenceMetric`、行动层的 `ToolCorrectnessMetric` 与 `ArgumentCorrectnessMetric`、以及轨迹级执行质量的 `TaskCompletionMetric` 与 `StepEfficiencyMetric` 入手。规划与执行指标需要追踪、对完整有序轨迹打分；工具指标仍是组件级。

**Q：端到端、轨迹、组件级 Agent 评测有何区别？**
A：端到端评测对最终黑盒输出打分，不检查内部执行。轨迹评测使用追踪、对完整有序轨迹打分——用于计划质量、计划遵循度、任务完成度、步骤效率。组件级评测通过 `@observe(metrics=[...])` 挂载、对特定 span（如 LLM 工具调用决策）打分——用于工具选择与参数正确性。

**Q：评测 AI Agent 一定需要追踪吗？**
A：轨迹与组件级 Agent 指标需要追踪，因为它们读取「完整有序轨迹」或「特定 span」。用 `@observe` 包裹 Agent 函数，轨迹就会自动构建。黑盒端到端输出评测不需要内部追踪。

**Q：我能写自定义 AI Agent 评测指标吗？**
A：能。用 `GEval` 处理「推理清晰度、专业语气」这类主观自然语言标准，用 `DAGMetric` 处理确定性决策树逻辑。自定义指标应用于「其测试用例与评测参数所支持的范围」。

**Q：如何在生产中运行 AI Agent 评测？**
A：用 DeepEval 在本地运行开发评测，然后把轨迹导出到 [Confident AI](https://www.confident-ai.com/products/llm-observability)，它会浮现失败的生产轨迹，让你把它们转化为下一轮评测的 golden。

---

## 下一步与更多资源

如果你已在生产环境，[Confident AI](https://www.confident-ai.com/products/llm-observability) 能在还没人为失败写评测之前就浮现它们、聚类成失败模式、路由给合适的标注者，并转化为下一轮评测的 golden。接下来：

1. **理解轨迹评测**——在[基于轨迹的评测指南](/docs/evaluation-trajectory-based-llm-evals)学习如何对完整有序 Agent 轨迹打分。
2. **探索指标**——在 [AI Agent 评测指标指南](/guides/guides-ai-agent-evaluation-metrics)学习每个指标如何工作（含计算公式与配置项）。
3. **阅读完整指南**——关于 Agent 评测如何与语音 AI、RAG、MCP 并列，以及开发/生产最佳实践，阅读 [LLM 评测：实用指南](/blog/ultimate-llm-evaluation-guide)。
4. **接入生产轨迹**——在终端运行 `deepeval login`，开始向 Confident AI 发送轨迹。
5. **加入社区**——有问题？加入 [DeepEval Discord](https://discord.com/invite/a3K9c8GRGt)。

**恭喜 🎉！** 你现在已具备为 AI Agent 构建健壮评测流水线的知识。