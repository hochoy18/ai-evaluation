# AI 评估框架选型与学习路径研究报告

> 调研日期:2026-11-17。所有版本、维护状态、代码示例均于当日通过官方文档、GitHub 仓库(API/源码)与 arXiv 原文核实;每个关键论断附一手来源链接。
> 读者背景:已在学习 Langfuse(观测/评估平台),目标是上手通用 AI Agent / RAG 评估框架,并同步学习评估理论。

---

## TL;DR:从哪个框架开始?

**第一推荐:DeepEval。** 一句话理由:它是唯一一个用"pytest 式单元测试"的心智模型、在本地 5 分钟跑通、且同时内置 RAG 五指标 + Agent 六指标 + G-Eval 自定义 judge 的框架——每个指标的官方文档都带公式和算法拆解([How Is It Calculated](https://deepeval.com/docs/metrics-faithfulness)),框架本身就是理论教材,且不强制绑定任何云平台([官方 quickstart](https://deepeval.com/docs/getting-started))。

配套路线:**DeepEval 上手 → 读 Ragas 论文和指标源码补 RAG 理论 → 读 MT-Bench / G-Eval 论文补 judge 理论 → 用 Inspect AI 进入严格的 Agent 基准评估 → 把分数接回你已熟悉的 Langfuse 做在线评估与实验管理。** 详见[第 6 节学习路径](#6-分阶段学习路径)。

三个重要的"避坑"事实(均为 2026 年核实):

1. **HELM 已于 2026-06-01 进入维护模式**(官方 README 明示),不适合作为 2026 年的入门框架。来源:[HELM README](https://github.com/stanford-crfm/helm)。
2. **OpenAI Evals 开源仓库已低活跃**(最后一次功能性提交是 2025-11,2026-04 的提交仅为安全加固),OpenAI 的重心已转向 Dashboard 里的平台化 Evals 产品。来源:[openai/evals README](https://github.com/openai/evals)、[提交记录](https://github.com/openai/evals/commits/main)。
3. **Ragas 仓库已从 `explodinggradients/ragas` 迁移到 `vibrantlabsai/ragas`**,且 0.4 版正在把旧 `evaluate()` API 迁移到新的 collections API(旧 API 将在 1.0 移除)——网上大量旧教程的 import 路径已过时。来源:[GitHub 重定向](https://github.com/explodinggradients/ragas)、[Ragas faithfulness 文档的 Deprecation Timeline](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)。

---

## 1. 先分清三类东西:评估框架、基准、评估平台

学评估之前先建立这张概念地图,后面所有工具都能对号入座:

| 类别 | 是什么 | 回答的问题 | 代表 |
|---|---|---|---|
| **评估框架 (eval framework)** | 可复用的代码库:定义 dataset / metric / judge 的组织方式,让你对**自己的应用**写评估 | "我的 RAG/Agent 在这批测试用例上表现如何?" | DeepEval、Ragas、Inspect AI、promptfoo、TruLens |
| **基准 (benchmark)** | 固定的任务集 + 评分协议,用于**横向比较模型/系统** | "GPT/Claude/开源模型谁更强?" | MMLU、GAIA、τ-bench、AgentBench、SimpleQA |
| **评估平台 (eval platform)** | 存储与协作基础设施:数据集管理、实验对比、在线打分、人工标注、看板 | "分数存在哪?怎么对比版本?怎么监控生产?" | Langfuse、LangSmith、Arize Phoenix |

三者是互补关系,不是竞争关系。典型组合:**框架算分 → 平台存分/管实验 → 基准提供公开可比的任务集**。例如 Langfuse 官方文档明确描述了这种分工:"Ragas defines and computes the metric, Langfuse stores the score next to the trace that produced the answer. You do not have to choose between them."([Langfuse × Ragas 集成文档](https://langfuse.com/integrations/frameworks/ragas))。Inspect AI 则同时横跨框架与基准:它本身是框架,又内置了 200+ 个现成基准实现([Inspect 官方首页](https://inspect.aisi.org.uk/))。

另一个贯穿全文的核心区分:**模型级评估**(测底座模型能力,如 lm-evaluation-harness、HELM)vs **应用级评估**(测你搭的 RAG/Agent 系统,如 DeepEval、Ragas)。你的目标属于后者,前者了解即可。

---

## 2. 框架对比总表

版本/星数/license 均于 2026-11-17 通过 GitHub API 核实。

| 框架 | 定位 | 核心抽象 | 代表性内置指标 | 平台绑定 | License | 最新版本(发布日) | 上手成本 |
|---|---|---|---|---|---|---|---|
| **DeepEval** | RAG + Agent + 多轮 + 安全,应用级 | `LLMTestCase` / `Metric` / pytest 式 `assert_test`;Agent 用 `@observe` trace + `evals_iterator` | Faithfulness、AnswerRelevancy、Contextual Precision/Recall、TaskCompletion、ToolCorrectness、G-Eval | 无(Confident AI 可选) | Apache-2.0 | python-v4.2.4(2026-09-22) | ★ 极低,pip 装完即跑 |
| **Ragas** | RAG 专精 + 测试集生成,应用级 | `EvaluationDataset` / metric collections / `evaluate()` | Faithfulness、AnswerRelevancy、ContextPrecision/Recall、FactualCorrectness、ToolCallAccuracy | 无 | Apache-2.0 | v0.4.4(2026-11-14) | ★ 低,但 0.4 新旧 API 并存 |
| **Inspect AI** | 通用 LLM/Agent 评估 + 200 多个现成基准 | `Task = Dataset + Solver + Scorer`;`@task`;sandbox;`react()` agent | includes/match/exact/f1/choice/math、model_graded_qa/fact | 无(UK AISI 官方) | MIT | 0.3.290(2026-11-14) | ★★ 中,概念清晰但体系大 |
| **promptfoo** | 声明式 prompt/Agent/RAG 测试 + 红队 | YAML config:prompts × providers × tests × assert | llm-rubric、g-eval、factuality、context-faithfulness、trajectory:goal-success | 无 | MIT | 0.125.1(2026-10-25) | ★ 低,npx 一条命令 |
| **TruLens** | RAG/Agent 评估 + OTel 追踪 | `TruApp` + `Metric`(feedback function)+ `Selector` | RAG Triad:Groundedness、Context Relevance、Answer Relevance;Agent GPA | 无(本地 dashboard) | MIT | trulens-2.16.0(2026-10-19) | ★★ 中,需先 instrument |
| **Phoenix (evals)** | 观测平台 + 独立 evals 库 | DataFrame + `Evaluator` + `LLM`/`DecisionModel` judge | Faithfulness、Correctness、RetrievalRelevance、Conciseness | evals 库可独立用;看板需 Phoenix | **Elastic License 2.0**(非 OSI 开源) | evals 库 3.x(2026-11 活跃) | ★ 低(仅 evals 库) |
| **lm-evaluation-harness** | 底座模型学术基准 | Task YAML + LM 后端 + filter/metric | 60+ 学术基准、数百子任务(MMLU 等) | 无 | MIT | v0.4.13(2026-08-31) | ★★ 中,偏模型级 |
| **HELM** | 底座模型"全面评估"(已维护模式) | scenario × model × metric | 准确率之外的 efficiency/bias/toxicity 等 | 无 | Apache-2.0 | v0.5.16(2026-04-30) | ★★★ 高,**不建议入门** |
| **OpenAI Evals** | 评估注册表 + YAML 模板(低活跃) | eval registry(YAML)+ completion function | model-graded 模板、match 等 | 倾向 OpenAI 平台 | MIT | 无 release;功能提交止于 2025-11 | ★★ 中,不建议入门 |
| **Langfuse** | 评估平台(你已在学) | Score / Evaluator / Dataset / Experiment | LLM-as-judge、decision model、code evaluator、人工标注 | 自身即平台;MIT 核心 + ee 目录商业许可 | MIT(+EE) | v4.81.0(2026-11-16) | 你已会 |
| **LangSmith** | 评估平台(LangChain 官方) | Dataset / Evaluator / Experiment / Annotation Queue | LLM-judge、pairwise、code、人工 | 绑定 LangSmith 云服务(闭源平台) | 专有 | 持续更新 | ★ 低(若用 LangChain) |

来源:各仓库 GitHub API 元数据与官方文档,详见[第 3 节](#3-各框架详评)逐项引用与[参考列表](#7-参考来源)。

---

## 3. 各框架详评

### 3.1 DeepEval —— 第一推荐

- **仓库**:[confident-ai/deepeval](https://github.com/confident-ai/deepeval),约 19.5k stars,Apache-2.0,最近 push 2026-11-16;Python 版 v4.2.4(2026-09-22),另有 TypeScript 版(接入 Vitest)。
- **定位**:README 自述 "similar to Pytest but specialized for unit testing LLM apps",并强调 "incorporates the latest research … run **locally on your machine**"([README](https://github.com/confident-ai/deepeval))。覆盖 RAG、Agent、聊天机器人、多模态、安全五大类指标([metrics-introduction](https://deepeval.com/docs/metrics-introduction))。
- **核心抽象**:
  - `LLMTestCase`(input / actual_output / expected_output / retrieval_context / tools_called 等)= 一次应用交互;
  - `Metric`(threshold + measure → score + reason)= 一个评分器,多数是 self-explaining LLM-Eval,输出分数同时输出理由;
  - `assert_test` / `deepeval test run` = pytest 式断言与运行器,天然进 CI;
  - Agent 评估用 `@observe()` 装饰器生成 trace,`dataset.evals_iterator(metrics=[...])` 把整条 trajectory 交给指标([trajectory-based evals](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals))。
- **内置指标**(官方文档核实):
  - RAG:Answer Relevancy、Faithfulness、Contextual Relevancy / Recall / Precision;
  - Agent **trajectory 级**:Task Completion、Step Efficiency、Plan Adherence、Plan Quality;**组件级**:Tool Correctness、Argument Correctness;
  - 多轮:Conversation Completeness、Conversation Relevancy、Role Adherence、Knowledge Retention;
  - 自定义:**G-Eval**(自然语言 criteria → CoT 评估步骤 → 打分,官方文档直接引用 [G-Eval 论文](https://arxiv.org/abs/2303.16634)并说明默认自动做 token 概率加权归一化)、DAG 指标、以及新的 JevEval("System One" judge)等([G-Eval 文档](https://deepeval.com/docs/metrics-llm-evals))。
- **上手成本**:`pip install -U deepeval` + 一个 test 文件 + `deepeval test run`。官方明确 "Everything in this quickstart runs locally",Confident AI 云平台只是可选项(`deepeval login` 才接入)([getting-started](https://deepeval.com/docs/getting-started))。依赖轻,不强制任何平台。
- **文档质量**:极高。每个指标页有 Required Arguments / How Is It Calculated(带公式)/ Customize Your Template / FAQs,等于把论文算法翻译成工程文档。
- **与 Langfuse 衔接**:Langfuse 官方集成文档给出两种模式——生产 trace 批量打分(`langfuse.api.observations.get_many()` + `create_score()`)和数据集实验(`dataset.run_experiment(evaluators=[...])`),DeepEval 的 reason 写入 score comment,在 Langfuse UI 里可解释([Langfuse × DeepEval](https://langfuse.com/resources/engineering/deepeval))。注意:Langfuse 内置 code evaluator 只能跑标准库,DeepEval 要在你自己的进程里跑再回写分数(同文档说明)。

**官方 quickstart 最小示例**(摘录自 [deepeval.com/docs/getting-started](https://deepeval.com/docs/getting-started),Python):

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
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output from your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])
```

运行:`deepeval test run test_example.py`(需先设置 `OPENAI_API_KEY`,或按文档配置其他 judge 模型)。

RAG 场景则换成专用指标,同样三行核心逻辑(摘录自 [metrics-faithfulness](https://deepeval.com/docs/metrics-faithfulness)):

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(threshold=0.7, model="gpt-4.1", include_reason=True)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

### 3.2 Ragas —— RAG 指标理论的"原产地",第二步必学

- **仓库**:[vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas)(原 explodinggradients/ragas,GitHub 已永久重定向),约 16.3k stars,Apache-2.0,v0.4.4 发布于 2026-11-14(GitHub 与 PyPI 一致),维护活跃。
- **定位**:RAG 评估的事实标准之一,论文 [Ragas: Automated Evaluation of Retrieval Augmented Generation(arXiv:2309.15217)](https://arxiv.org/abs/2309.15217) 提出**无需人工标注 ground truth 的 reference-free RAG 评估**。除指标外还提供测试集自动生成(testset generation)([README](https://github.com/vibrantlabsai/ragas))。
- **核心抽象**:样本(`user_input` / `retrieved_contexts` / `response` / `reference`)→ `EvaluationDataset` → `evaluate(dataset, metrics, llm)`。**注意 0.4 的 API 迁移**:官方文档现在推荐 collections 风格(`ragas.metrics.collections` + `llm_factory` + `scorer.ascore()`),旧 `SingleTurnSample`/`evaluate()` 风格标注为 "deprecated in 0.4, removed in 1.0"([faithfulness 文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/))。入门教程页目前仍用旧风格,读文档时留意区分。
- **内置指标**:Faithfulness、Answer Relevancy、Context Precision(含 with/without reference、Non-LLM、ID-based 变体)、Context Recall、Factual Correctness、Answer Correctness、Noise Sensitivity、Summarization Score,以及 Agent 向的 Topic Adherence、Tool call Accuracy(支持 flexible order 模式)([available metrics 目录](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/))。每个指标页都有公式 + 手算示例,是学 RAG 评估理论最好的免费教材。
- **上手成本**:`pip install ragas`,准备 evaluator LLM(+部分指标需要 embedding),几十行代码。不绑定平台。
- **与 Langfuse 衔接**:Langfuse 有官方集成页,把 Ragas 指标包装成 experiment evaluator;Langfuse 的托管评估器目录(managed evaluators)也是与 Ragas 合作构建的([集成文档](https://langfuse.com/integrations/frameworks/ragas)、[Langfuse agent evaluation 指南](https://langfuse.com/resources/engineering/ai-agent-evaluation))。

**官方 quickstart 核心代码**(摘录自 [docs.ragas.io RAG eval 教程](https://docs.ragas.io/en/stable/getstarted/rag_eval/),数据收集部分略):

```python
# 每条样本:问题 + 检索到的上下文 + 应用回答 + 参考答案
dataset = []
for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append({
        "user_input": query,
        "retrieved_contexts": relevant_docs,
        "response": response,
        "reference": reference,
    })

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

evaluator_llm = LangchainLLMWrapper(llm)   # llm = ChatOpenAI(model="gpt-4o")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)
# 输出示例:{'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}
```

0.4 新 collections API 的单指标用法(摘录自 [faithfulness 文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)):

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
scorer = Faithfulness(llm=llm)
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=["The First AFL–NFL World Championship Game was ... played on January 15, 1967 ..."],
)
print(result.value)  # 1.0
```

### 3.3 Inspect AI —— Agent/基准评估的"正规军",进阶首选

- **仓库**:[UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai),约 3.3k stars,MIT,0.3.290(2026-11-14),几乎每日提交。由 **UK AI Security Institute(英国 AI 安全研究所)与 Meridian Labs** 开发([官方文档首页](https://inspect.aisi.org.uk/))。
- **定位**:前沿模型评估框架,覆盖 coding、agentic tasks、reasoning、knowledge、多模态;内置 **200+ 现成基准实现**、20+ 模型提供商、Docker/Kubernetes 等沙箱、VS Code 扩展与 Web 日志查看器(`inspect view`)(同上)。
- **核心抽象**:`Task = Dataset(input/target 样本)+ Solver(产生答案,可以是单次 generate 也可以是完整 agent)+ Scorer(打分)`。这个三元组是理解一切评估框架的最佳心智模型。
- **内置 scorer**:`includes()`、`match()`、`exact()`、`pattern()`、`answer()`、`f1()`、`choice()`、`math()`(SymPy 数学等价比较)、`perplexity()`,以及 model-graded 的 `model_graded_qa()` / `model_graded_fact()`;默认报告 accuracy 与 stderr([scorers 文档](https://inspect.aisi.org.uk/scorers.html))。
- **Agent 支持**(这是它相对其他框架的差异化):内置 `react()` agent、长程 Deep Agent、multi-agent `handoff()`、Agent Bridge 接入 OpenAI Agents SDK / LangChain / Pydantic AI 外部 agent、Human Agent 做人类基线,还有 checkpointing、intervention、token/时间 limits([agents 文档](https://inspect.aisi.org.uk/agents.html))。
- **上手成本**:`pip install inspect-ai`,写一个 `@task` 函数,`inspect eval xxx.py --model openai/...`。概念清晰,但沙箱、solver 组合等体系较大,适合第二阶段。不绑定任何平台。
- **文档质量**:极高,且提供 llms.txt / llms-guide.txt 供 LLM 辅助学习([官方首页](https://inspect.aisi.org.uk/))。

**官方 Hello World**(摘录自 [inspect.aisi.org.uk](https://inspect.aisi.org.uk/)):

```python
# simpleqa.py —— 问答基准:dataset + solver + scorer 三元组
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

@task
def simpleqa():
    return Task(
        dataset=hf_dataset(
            "codelion/SimpleQA-Verified",
            split="train",
            sample_fields=FieldSpec(input="problem", target="answer"),
        ),
        solver=generate(),            # 直接把 input 发给模型
        scorer=model_graded_qa(),     # 用 judge 模型对照 target 打分
    )
```

```bash
inspect eval simpleqa.py --model openai/gpt-6-luna
inspect view   # 浏览器查看结果与逐条 transcript
```

```python
# ctf.py —— Agent 评估:ReAct agent + 工具 + Docker 沙箱
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, todo_write

@task
def ctf():
    return Task(
        dataset=json_dataset("challenges.json"),
        solver=react(
            prompt="You are a Capture the Flag player. Explore the system and find the flag.",
            tools=[bash(), todo_write()],
            attempts=3,
        ),
        scorer=includes(),     # 目标 flag 出现在提交答案中即通过
        sandbox="docker",      # bash 工具在隔离容器里执行
    )
```

### 3.4 promptfoo —— 声明式 YAML 派,CI/红队场景的替代选择

- **仓库**:[promptfoo/promptfoo](https://github.com/promptfoo/promptfoo),约 27.3k stars(本表最高之一),MIT,0.125.1(2026-10-25),每日提交。仓库描述:"Test your prompts, agents, and RAGs. Red teaming/pentesting/vulnerability scanning for AI … Used by OpenAI and Anthropic"。
- **核心抽象**:YAML 声明 `prompts × providers × tests`,每个 test 挂 `assert`(确定性断言 + model-graded 断言);`npx promptfoo eval` 跑,`npx promptfoo view` 看对比矩阵。
- **内置 model-graded 指标**:llm-rubric、g-eval、factuality、answer-relevance、context-faithfulness、context-recall、context-relevance、select-best、model-graded-closedqa 等([model-graded metrics 文档](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/))。
- **Agent 评估**:有 `trajectory:goal-success`(基于 trace 数据让 judge 判断任务是否达成)以及确定性的 `trajectory:tool-used` / `trajectory:tool-args-match` / `trajectory:tool-sequence` 断言(同上文档)。
- **RAG 评估**:官方指南强调把检索与生成两步分开评,检索步可用 `file://retrieve.py` provider 单独测([evaluate-rag 指南](https://www.promptfoo.dev/docs/guides/evaluate-rag/))。
- **上手成本**:极低,`npx promptfoo@latest init --example getting-started` 一条命令生成可跑示例([getting-started](https://www.promptfoo.dev/docs/getting-started/))。Node 生态,Python 团队略有摩擦。
- **为什么不是第一推荐**:指标算法的"理论透明度"不如 DeepEval/Ragas(YAML 背后是内置 rubric prompt,可覆写但文档不逐条讲算法);其重心近年明显偏向红队/安全扫描。适合你之后做 CI 门禁与多模型对比时引入。

### 3.5 TruLens —— RAG Triad 的另一出处,OTel 原生

- **仓库**:[truera/trulens](https://github.com/truera/trulens),约 3.7k stars,MIT,trulens-2.16.0(2026-10-19),活跃。
- **定位**:README 自述 "TruLens finds where your agent fails and where you can cut cost without losing quality. Open source, OpenTelemetry-native."——评估 + 追踪一体,trace 可移植到任何 OTLP 后端([README](https://github.com/truera/trulens))。
- **核心抽象**:`@instrument` 装饰器标注 RETRIEVAL/GENERATION 等 span → `Metric`(feedback function,由 provider 方法 + `Selector` 选择器组成)→ `TruApp(app, feedbacks=[...])` 包裹应用 → 运行时自动打分 → `session.get_leaderboard()` / `run_dashboard()` 本地看板。
- **标志性理论贡献**:**RAG Triad**(Groundedness / Context Relevance / Answer Relevance,quickstart 中称 "hallucination triad"),与 Ragas 三元组互为印证([官方 quickstart notebook](https://github.com/truera/trulens/blob/main/examples/quickstart/quickstart.ipynb)、[RAG Triad 概念页](https://www.trulens.org/getting_started/core_concepts/rag_triad/))。README 还宣称其 Agent GPA judge 在 TRAIL/GAIA 人工标注错误上捕获率 95%(引 [arXiv:2510.08847](https://arxiv.org/abs/2510.08847),此为厂商自述,供参考)。
- **上手成本**:中等——必须先 instrument 应用才能评,quickstart 依赖 chromadb 等;适合想要"评估+追踪一体、OTel 标准化"的团队。与 Langfuse 功能重叠度高(都是观测+评估),二者选一即可,你已学 Langfuse,故 TruLens 优先级降低。

### 3.6 Arize Phoenix(evals 部分)—— 平台型,evals 库可独立用

- **仓库**:[Arize-ai/phoenix](https://github.com/Arize-ai/phoenix),约 12.3k stars,**Elastic License 2.0(ELv2)**——允许自托管使用,但禁止拿它做托管 SaaS 竞品([LICENSE](https://github.com/Arize-ai/phoenix/blob/main/LICENSE)),严格说不是 OSI 开源 license,选型时注意。
- **定位**:"open-source AI observability platform designed for experimentation, evaluation, and troubleshooting",OTel/OpenInference 追踪,厂商/语言无关([README](https://github.com/Arize-ai/phoenix))。
- **evals 核心抽象**:`arize-phoenix-evals` 3.x 独立库——pandas DataFrame + `LLM(provider, model)` judge + `Evaluator`,`async_evaluate_dataframe(df, evaluators=[...], concurrency=10)` 批量打分;另有 `DecisionModel`(返回标签+概率的结构化 judge)。内置 FaithfulnessEvaluator、CorrectnessEvaluator、RetrievalRelevanceEvaluator、ConcisenessEvaluator 等;结果可选回写 Phoenix 看板([官方 evals quickstart](https://arize.com/docs/phoenix/evaluation/evals))。
- **上手成本**:仅用 evals 库很低(pip 两个包);完整平台要部署 Phoenix。与 Langfuse 定位重叠,作为了解项即可。

### 3.7 lm-evaluation-harness(EleutherAI)—— 模型级基准的事实标准

- **仓库**:[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),约 14.5k stars,MIT,v0.4.13(2026-08-31)。
- **定位**:底座模型的 few-shot 学术基准评估,60+ 标准基准、数百子任务;是 Hugging Face Open LLM Leaderboard 的后端,被 NVIDIA、Cohere 等内部使用([README](https://github.com/EleutherAI/lm-evaluation-harness))。
- **近期变化**(README Latest News):2025-12 CLI 重构为 `run/ls/validate` 子命令 + YAML 配置;基础包不再捆绑 torch/transformers,后端按需装(`lm_eval[hf]`、`lm_eval[vllm]`);2026-09 新增插件机制(entry point 注册模型后端/filter/metric)。
- **与你的目标的关系**:它评的是"模型"而不是"你的 RAG/Agent 应用",指标以 exact match / loglikelihood / accuracy 为主。学评估理论时值得跑一次 MMLU 子集感受模型级评估,但不必深入。

### 3.8 HELM(Stanford CRFM)—— 理论重要,但已进入维护模式

- **仓库**:[stanford-crfm/helm](https://github.com/stanford-crfm/helm),约 3.0k stars,Apache-2.0,v0.5.16(2026-04-30)。
- **关键状态**:README 顶部明示 "**HELM entered maintenance mode on June 1, 2026**"([README](https://github.com/stanford-crfm/helm)、[维护模式政策](https://crfm-helm.readthedocs.io/en/latest/maintenance_mode/))。
- **理论价值**:HELM 论文([TMLR](https://openreview.net/forum?id=iO4LZibEqW))提出"全面评估"方法论——不止 accuracy,还测 efficiency、bias、toxicity、calibration 等多维度,并强调可复现与透明。这套"多场景 × 多指标"思想值得读论文吸收;框架本身不建议 2026 年入门。

### 3.9 OpenAI Evals —— 历史地位高,现状低活跃

- **仓库**:[openai/evals](https://github.com/openai/evals),约 19.8k stars,MIT([LICENSE.md](https://github.com/openai/evals/blob/main/LICENSE.md)),无正式 release;最后一次功能性提交 2025-11-03(移除失效依赖),2026-04 的两次提交仅为 CI/pre-commit 安全加固([提交历史](https://github.com/openai/evals/commits/main))。
- **现状**:README 第一行即引导用户去 OpenAI Dashboard 的平台化 Evals 产品([platform.openai.com/docs/guides/evals](https://platform.openai.com/docs/guides/evals));开源部分主要是 eval registry(YAML 模板 + Git-LFS 数据)与 completion function 协议([README](https://github.com/openai/evals))。
- **结论**:作为"评估注册表"的历史设计值得了解,但不适合作为学习主线。

### 3.10 平台型:Langfuse(你已在学)与 LangSmith

**Langfuse**([langfuse/langfuse](https://github.com/langfuse/langfuse),约 37.5k stars,v4.81.0,2026-11-16):

- 2026-01-16 被 ClickHouse 收购,官方承诺保持开源与自托管、license 不变([Langfuse 官方博客](https://langfuse.com/blog/joining-clickhouse)、[ClickHouse 官方公告](https://clickhouse.com/blog/clickhouse-acquires-langfuse-open-source-llm-observability))。核心 MIT,`ee/` 目录为商业许可([LICENSE](https://github.com/langfuse/langfuse/blob/main/LICENSE))。
- 评估体系(官方文档核实):
  - **Score** 是统一数据对象(NUMERIC/CATEGORICAL/BOOLEAN/TEXT),可挂在 trace/observation/session/dataset run 上;
  - **评估方法**:LLM-as-a-Judge、Decision-model evaluator、Code evaluator(在 Langfuse 内执行、仅标准库)、Annotation Queues 人工标注、SDK/UI 打分;
  - **离线**:Dataset + Experiment(UI / SDK / OpenTelemetry 三种跑法),CI 里用 `langfuse/experiment-action` GitHub Action 做回归门禁;
  - **在线**:Evaluator + Rule(过滤 + 采样率)对生产 observation 实时打分,支持历史回填;
  - **Score Analytics**:内置 Pearson 相关、**Cohen's Kappa**、F1、Overall Agreement,用于校验 judge 与人工标注的一致性([core-concepts](https://langfuse.com/docs/evaluation/core-concepts)、[overview](https://langfuse.com/docs/evaluation/overview)、[score-analytics](https://langfuse.com/docs/evaluation/scores/score-analytics))。
- 官方还有专门的 [AI agent evaluation 工程指南](https://langfuse.com/resources/engineering/ai-agent-evaluation)(四维度拆解,见第 5.3 节)与 [Ragas](https://langfuse.com/integrations/frameworks/ragas)/[DeepEval](https://langfuse.com/resources/engineering/deepeval) 集成文档——这正是"框架算分、平台存分"的落地路径。

**LangSmith**([docs.langchain.com/langsmith/evaluation](https://docs.langchain.com/langsmith/evaluation)):LangChain 官方闭源云平台。概念与 Langfuse 高度同构:离线(dataset + evaluator + experiment)与在线(production evaluator + 采样)两条线;evaluator 支持 human review / code rules / LLM-as-judge / **pairwise comparison**;强调"把生产失败 trace 变成 dataset 里的永久回归测试"的反馈闭环。若你的应用基于 LangChain/LangGraph 可考虑,否则与 Langfuse 二选一,不必重复学。

### 3.11 Agent 基准速览:AgentBench、τ-bench、GAIA

基准 ≠ 框架:它们是固定任务集 + 评分协议,用来横向比较;框架用来评你自己的系统。

| 基准 | 内容 | 评分方式 | 来源 |
|---|---|---|---|
| **AgentBench**(ICLR'24) | 8 个交互环境(OS、DB、知识图谱、卡牌游戏等)测 LLM-as-Agent 的推理与决策 | 各环境任务成功率 | [论文 arXiv:2308.03688](https://arxiv.org/abs/2308.03688)、[THUDM/AgentBench](https://github.com/THUDM/AgentBench) |
| **τ-bench**(Sierra) | LLM 模拟用户 + 领域 API 工具 + 政策约束的动态对话(航空/零售客服) | **对比对话结束时的数据库状态与标注目标状态**(结果导向,不靠 judge 读文本);提出 **pass^k** 衡量多次试验的可靠性 | [论文 arXiv:2406.12045](https://arxiv.org/abs/2406.12045)、[sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) |
| **GAIA**(Meta 等) | 对人类概念上简单、对 AI 难的真实问题:推理、多模态、网页浏览、工具使用 | 答案精确匹配;论文报告人类 92% vs GPT-4+插件 15% | [论文 arXiv:2311.12983](https://arxiv.org/abs/2311.12983) |

**重要更新**:τ-bench 原仓库 README 顶部警告任务已过时,最新演进在 [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench):τ²-bench([arXiv:2506.07982](https://arxiv.org/abs/2506.07982))新增 telecom 域与双工语音,现已升级为 **τ³-bench**(新增 banking 域、六语言 i18n、intake 域、post-hoc judges)。τ-bench 论文的关键发现值得记住:即使 gpt-4o 级 function-calling agent,任务成功率也不到 50%,retail 域 pass^8 < 25%——**agent 的"可靠性"必须多次试验测量,单次通过不算数**。

---

## 4. 评估理论必读(每个概念附一手来源)

### 4.1 经典 NLP 指标为什么不够用:BLEU / ROUGE / exact match

- BLEU([Papineni et al., 2002, ACL](https://aclanthology.org/P02-1040/))与 ROUGE([Lin, 2004, ACL](https://aclanthology.org/W04-1013/))本质是 **n-gram 重叠统计**:前者为机器翻译设计(精度导向),后者为摘要设计(召回导向)。
- 局限的一手证据:G-Eval 论文开篇即指出 "Conventional reference-based metrics, such as BLEU and ROUGE, have been shown to have **relatively low correlation with human judgments**, especially for tasks that require creativity and diversity"([arXiv:2303.16634](https://arxiv.org/abs/2303.16634))。
- 直觉理解:开放式回答有无数种正确表述,n-gram 重叠惩罚合法的改写;且它们需要参考答案(reference-based),而生产流量没有参考答案。这就是 LLM 时代转向 judge 模型的根本原因。exact match / F1 并未过时——Inspect 的 `exact()`/`f1()`、lm-eval-harness 仍大量使用,适用于**答案唯一可判定**的任务(数学、选择题、抽取式 QA)([Inspect scorers](https://inspect.aisi.org.uk/scorers.html))。

### 4.2 RAG 三元组/四元组指标

**理论出处**:Ragas 论文([arXiv:2309.15217](https://arxiv.org/abs/2309.15217))提出 reference-free 地评估 RAG 的三个维度:检索质量(context relevance)、生成忠实度(faithfulness)、回答质量(answer relevance)。TruLens 独立提出同构的 "RAG Triad":Groundedness / Context Relevance / Answer Relevance([TruLens 概念页](https://www.trulens.org/getting_started/core_concepts/rag_triad/))。工程上常扩展为四元组(+ context precision/recall 拆分检索的排序与覆盖)。

以下定义与公式均摘自 Ragas 官方指标文档(2026-11 核实):

| 指标 | 测什么 | 怎么算 | 需要 reference? |
|---|---|---|---|
| **Faithfulness** | 回答是否忠于检索上下文(生成端幻觉) | 把 response 拆成 claims,逐条判断能否由 retrieved_contexts 推出;`F = 被支持的 claims 数 / 总 claims 数` | 否(reference-free) |
| **Answer Relevancy** | 回答是否切题(不评事实对错) | 由 response 反向生成 N 个问题(默认 3,`strictness` 可调),求这些问题 embedding 与原问题 embedding 的**平均余弦相似度** | 否(需 embedding) |
| **Context Precision** | 检索排序质量:相关 chunk 是否排在前面 | `Precision@k` 的加权平均,权重 v_k 为该位置是否相关;**对顺序敏感**——无关 chunk 排第一时分数从 ~1.0 掉到 ~0.5(官方示例) | 有 with/without reference 两版 |
| **Context Recall** | 检索覆盖:参考答案所需信息是否都被检索到 | 把 reference 拆成句子/claims,判断每条能否归因到 retrieved_contexts | 是(需 reference 或 reference_contexts) |

来源:[Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)、[Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)、[Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)、[Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)。DeepEval 的同名指标算法略有差异(如 Faithfulness 定义为"不与上下文矛盾的 claims 占比",并输出 reason),对照阅读很有启发:[DeepEval Faithfulness](https://deepeval.com/docs/metrics-faithfulness)。

**诊断逻辑**(为什么拆维度而不是一个总分):faithfulness 低而 context 相关 → 生成端问题,改 prompt;context 不相关 → 检索端问题,改索引/embedding。Langfuse 的 RAG agent 评估指南把这套逻辑落到了 trace 的不同 observation 上([来源](https://langfuse.com/resources/engineering/ai-agent-evaluation))。

### 4.3 LLM-as-a-judge:原理、偏差与校准

**必读论文 1:MT-Bench / Chatbot Arena**([Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, arXiv:2306.05685](https://arxiv.org/abs/2306.05685),Zheng et al., NeurIPS 2023)。以下细节均核实自[论文原文](https://arxiv.org/html/2306.05685v4):

- **核心结论**:GPT-4 级 judge 与人类偏好的一致率超过 80%,达到人类之间的一致水平(基于 3K 专家投票 + 3K 众包投票验证)——这是"用模型评模型"合法性的奠基证据。
- **Position bias(位置偏差)**:judge 倾向 favor 第一个答案。交换两个答案顺序测一致性:GPT-4 默认 prompt 仅 65% 一致,GPT-3.5 46.2%,Claude-v1 23.8%。缓解:交换位置双向评判、检查一致性。
- **Verbosity bias(冗长偏差)**:judge 偏爱更长的回答,即使信息量相同。论文用 "repetitive list" 攻击(把列表改写重复一遍、不增加新信息)验证,当时只有 GPT-4 识破。
- **Self-enhancement bias(自我偏爱)**:judge 可能偏爱自己生成的答案;论文观察到 GPT-4 对自己胜率高约 10%、Claude-v1 约 25%,但作者注明数据有限、不宜过度解读。
- **推理能力局限**:judge 给数学/推理题打分不可靠;论文提出 **chain-of-thought judge** 与 **reference-guided judge**(给 judge 提供参考答案)两种缓解。
- 三种评判范式:pairwise comparison(两两对比)、single-answer grading(单答案打分)、reference-guided grading——对应你在各框架里看到的 select-best / llm-rubric / closedqa 等断言类型。

**必读论文 2:G-Eval**([G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment, arXiv:2303.16634](https://arxiv.org/abs/2303.16634),Liu et al., EMNLP 2023):

- 算法三步:① 给定任务描述与评估 criteria,让 LLM 生成 **CoT 评估步骤**;② 按步骤以"填表"方式对输出打分;③ 用输出 token 的**概率加权求和**归一化分数(`p(s) = Σ p(s_i)·s_i`),缓解整数打分的粗糙与偏差。
- 在摘要与对话生成任务上显著超越 BLEU/ROUGE 等传统指标与中等规模神经评估器,与人类判断相关性更高。
- 工程落地:DeepEval 的 `GEval` 是该论文最流行的实现,官方文档说明默认自动处理概率加权(自定义模型需实现返回 logprobs 的接口)([DeepEval G-Eval 文档](https://deepeval.com/docs/metrics-llm-evals));promptfoo 有 `g-eval` 断言([文档](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/))。

**校准(calibration)——judge 不是免检产品**:MT-Bench 的 80% 一致率是"对齐人类"的证据,你自己的 judge 也要对齐你的人工标注。Langfuse 内置了这件事的工具:Score Analytics 提供 **Cohen's Kappa、F1、Overall Agreement** 对比人工与 AI 标注([score-analytics](https://langfuse.com/docs/evaluation/scores/score-analytics)),官方还有 judge 校准指南([calibration skill](https://langfuse.com/guides/llm-as-a-judge-calibration-skill))。TruLens README 同样强调 "judges are graded against human annotations, out of the box"([来源](https://github.com/truera/trulens))。

### 4.4 Agent 评估:trajectory vs 最终结果

三个一手来源给出同构的方法论:

1. **DeepEval** 把评估分三个 scope:**end-to-end**(黑盒,只看 input/output)、**trajectory-based**(把完整有序执行链——plan、LLM 调用、tool call、子 agent handoff——作为一个整体评)、**component-level**(评单个 span,如某次检索或工具调用);并明确"当路径本身影响质量时用 trajectory 评估,三者可组合"([trajectory-based evals](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals))。对应指标:TaskCompletion/StepEfficiency/PlanAdherence/PlanQuality(trajectory)与 ToolCorrectness/ArgumentCorrectness(组件)([metrics-introduction](https://deepeval.com/docs/metrics-introduction))。
2. **Langfuse** 的 agent 评估指南提出四维度:**Trajectory**(路径是否合理:步数、冗余调用、循环)、**Tool use**(选对工具、参数合法、错误恢复)、**Task completion**(目标达成)、**Multi-turn**(会话级质量);并指出 agent 评估与 LLM 评估的本质差异——"评估单位是 trace 而不是 completion",中间步骤有独立失败模式,且工具错误常被最终答案掩盖(agent 自我修复了,但重试烧掉了 token 和时间)([AI agent evaluation](https://langfuse.com/resources/engineering/ai-agent-evaluation))。
3. **Inspect AI** 用 solver/agent 抽象直接跑 agent(react、多 agent、外部 agent bridge),scorer 既可以是结果导向(`includes()` 检查 flag)也可以是过程约束,配合 sandbox 保证工具执行安全([agents 文档](https://inspect.aisi.org.uk/agents.html))。

**结果导向评估的典范:τ-bench**——不靠 judge 读对话,而是**对比对话结束时的数据库状态与标注目标状态**,客观可判定;再用 **pass^k**(k 次独立试验全部成功的概率)度量可靠性([arXiv:2406.12045](https://arxiv.org/abs/2406.12045))。这提示你:能用确定性检查(代码/状态对比)就不要用 judge;judge 留给语义判断。Langfuse 指南的分层原则与此一致:code evaluator 全量跑(免费、毫秒级)→ LLM judge 跑语义子集 → 分歧/低置信样本进人工标注队列,人工标签再反过来校准 judge([来源](https://langfuse.com/resources/engineering/ai-agent-evaluation))。

### 4.5 离线 vs 在线评估;reference-based vs reference-free

- **离线(offline)**:发布前在固定 dataset 上跑 experiment,抓 prompt/模型/工具变更引起的回归;成本由数据集大小封顶。**在线(online)**:对生产 trace 按采样率持续打分,抓漂移、新输入、真实工具故障;发现的坏案例回灌 dataset 变成永久回归测试。Langfuse([core concepts](https://langfuse.com/docs/evaluation/core-concepts))与 LangSmith([evaluation 文档](https://docs.langchain.com/langsmith/evaluation))对这套闭环的描述几乎逐字同构——这是行业共识方法论,学一次到处适用。
- **reference-based**:有标准答案(expected_output / reference),如 Context Recall、factuality、closedqa;**reference-free**:无标准答案也能评,如 Faithfulness(只需上下文)、Answer Relevancy、Ragas 论文的核心卖点([arXiv:2309.15217](https://arxiv.org/abs/2309.15217))、DeepEval 指标页的 `referenceless` 标签([Faithfulness](https://deepeval.com/docs/metrics-faithfulness))。**生产在线评估几乎只能 reference-free**(真实用户问题没有标准答案),这是两阶段评估选型的关键约束。
- 同一指标常有双版本:Ragas Context Precision 有 with-reference(对照 reference 判 chunk 相关性)与 without-reference(对照 response 判,即 Context Utilization)两版([文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/))。

### 4.6 人工评估与标注一致性:Cohen's Kappa

- 人工评估是 ground truth 的来源,但标注者之间会有分歧。**Cohen's Kappa** 衡量两个标注者(或人工 vs judge)在**扣除随机一致后**的一致程度,取值 -1~1([Cohen, 1960 原始论文, DOI:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104))。
- 为什么不用裸一致率:两类标签、瞎猜也有 50% 一致率;kappa 把"碰巧一致"扣掉。Langfuse Score Analytics 对 categorical/boolean 分数内置 Kappa + F1 + Overall Agreement,并提供混淆矩阵视图([score-analytics](https://langfuse.com/docs/evaluation/scores/score-analytics))。
- 与 MT-Bench 的联系:该论文正是用"judge 与人类一致率 ≈ 人类之间一致率"来论证 judge 可替代部分人工([arXiv:2306.05685](https://arxiv.org/abs/2306.05685))。实践闭环:人工标注小批样本 → 算 judge 的 kappa → 不达标就改 judge prompt/换模型 → 达标后放量。

---

## 5. 为什么第一推荐是 DeepEval(而不是 Ragas / Inspect)

把候选放到你的四个约束下打分(均基于第 3 节核实的事实):

| 约束 | DeepEval | Ragas | Inspect | promptfoo |
|---|---|---|---|---|
| 上手最快(分钟级跑通) | ✅ 官方 5-min quickstart,pytest 心智模型 | ✅ 但 0.4 新旧 API 并存,教程易踩坑 | ⚠️ 概念清晰但体系大 | ✅ npx 即跑,但 Node 生态 |
| 同时覆盖 RAG + Agent | ✅ RAG 5 指标 + Agent 6 指标 + 多轮 | ⚠️ RAG 强,Agent 仅 2 个指标 | ✅ Agent 最强,RAG 专用指标少 | ⚠️ 两者都有但偏声明式 |
| 理论可学性(文档讲算法) | ✅ 每指标带公式 + 算法 + 论文引用 | ✅ 公式最详尽,但偏 RAG | ⚠️ 偏工程,理论靠论文 | ⚠️ rubric 可覆写但不讲算法 |
| 与 Langfuse 衔接 | ✅ 官方集成文档 | ✅ 官方集成文档 | ⚠️ 需自己写回分数 | ⚠️ 需自己写回分数 |
| 社区/维护 | ✅ 19.5k★,周更 | ✅ 16.3k★,周更 | ✅ 日更,官方机构背书 | ✅ 27.3k★,日更 |

结论:**DeepEval 是"上手速度 × 理论覆盖 × 双场景(RAG+Agent)"乘积最大的入口**;Ragas 是 RAG 理论的最佳补充读物(论文 + 公式文档),Inspect 是 Agent 评估的进阶终点。这不是"都可以"——按此顺序学,每一步都有明确产出。

---

## 6. 分阶段学习路径

> Python 环境按你的习惯用 uv 管理:`uv init eval-lab && cd eval-lab && uv add deepeval ragas openai`(官方文档写 pip,等价替换即可)。judge 模型建议先用便宜的小模型(gpt-4o-mini / gpt-5-nano 级)跑通,再换强模型对比——这本身就是一堂 judge 校准课。

### 阶段 1:跑通 DeepEval,建立"测试用例 × 指标 × judge"心智模型(1–2 天)

1. 跑官方 [5-min quickstart](https://deepeval.com/docs/getting-started)(GEval + assert_test + `deepeval test run`)。
2. 把 `actual_output` 换成你自己 RAG 应用的真实输出,跑 FaithfulnessMetric / AnswerRelevancyMetric / ContextualPrecisionMetric / ContextualRecallMetric([RAG 指标目录](https://deepeval.com/docs/metrics-introduction))。
3. 精读每个指标的 **How Is It Calculated** 小节,手算一遍官方示例(如 faithfulness 的 claims 拆分)。
4. 用 GEval 写一个你自己业务的自定义 criteria,观察它生成的 evaluation_steps——这就是 G-Eval 论文的 CoT 步骤生成。
5. 产出:一个能进 CI 的 `test_rag.py`,4+ 指标,全部带 reason。

### 阶段 2:对照 Ragas 与论文,补 RAG 评估理论(2–3 天)

1. 读 [Ragas 论文(arXiv:2309.15217)](https://arxiv.org/abs/2309.15217),重点:为什么 reference-free、三维度如何对应检索/生成两端。
2. 跑 [Ragas 官方 RAG eval 教程](https://docs.ragas.io/en/stable/getstarted/rag_eval/),对同一批数据对比 DeepEval 与 Ragas 同名指标的分数差异,读两边源码解释差异(DeepEval:`deepeval/metrics/`;Ragas:`src/ragas/metrics/`,仓库 [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas))。**同名指标算法不同**是评估领域最重要的实践认知之一。
3. 读 Ragas 的 [Context Precision 文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/),理解排序敏感性(with/without reference、Non-LLM、ID-based 变体)。
4. 顺手体验 Ragas testset generation(没有测试集时自动造题)。
5. 产出:一份"我的 RAG 系统体检报告",能区分检索端问题与生成端问题。

### 阶段 3:补 judge 理论(2 天,可与阶段 2 并行)

1. 精读 [MT-Bench 论文(arXiv:2306.05685)](https://arxiv.org/abs/2306.05685) 第 3 节:三大偏差 + 推理局限 + CoT/reference-guided 缓解。
2. 精读 [G-Eval 论文(arXiv:2303.16634)](https://arxiv.org/abs/2303.16634),对照 [DeepEval G-Eval 文档](https://deepeval.com/docs/metrics-llm-evals)看概率加权归一化的工程实现。
3. 动手实验:构造一对"内容相同、长度不同"的回答喂给你的 judge,复现 verbosity bias;交换顺序复现 position bias。
4. 人工标注 30–50 条样本,在 Langfuse 里用 [Score Analytics](https://langfuse.com/docs/evaluation/scores/score-analytics) 算 judge 与人工的 Cohen's Kappa,不达标就迭代 judge prompt([校准指南](https://langfuse.com/guides/llm-as-a-judge-calibration-skill))。
5. 产出:一个经过人工校准、偏差已知的 judge 配置。

### 阶段 4:Agent 评估(3–5 天)

1. 读 [Langfuse AI agent evaluation 指南](https://langfuse.com/resources/engineering/ai-agent-evaluation)(四维度 + trace-not-completion + 分层评估),这是最好的中文友好(英文但结构清晰)方法论总纲。
2. DeepEval:给你的 agent 加 `@observe()`,用 `evals_iterator` 跑 TaskCompletionMetric / ToolCorrectnessMetric / ArgumentCorrectnessMetric([trajectory 文档](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals))。
3. 读 [τ-bench 论文(arXiv:2406.12045)](https://arxiv.org/abs/2406.12045),理解状态对比评分与 pass^k;浏览 [AgentBench](https://arxiv.org/abs/2308.03688) 与 [GAIA](https://arxiv.org/abs/2311.12983) 摘要,建立基准视野。注意 τ-bench 已演进到 [τ³-bench](https://github.com/sierra-research/tau2-bench)。
4. 进阶到 **Inspect AI**:跑 [官方 Hello Inspect](https://inspect.aisi.org.uk/)(simpleqa + CTF 两个例子),然后从 [200+ 内置 evals](https://inspect.aisi.org.uk/evals/) 挑一个 agent 基准跑通;读 [agents](https://inspect.aisi.org.uk/agents.html) 与 [scorers](https://inspect.aisi.org.uk/scorers.html) 文档,理解 Task/Solver/Scorer 与 sandbox。
5. 产出:你的 agent 的 trajectory + 工具调用 + 任务达成三层评分;一个可复现的公开基准结果。

### 阶段 5:接回 Langfuse,形成生产闭环(持续)

1. 按 [Langfuse × DeepEval 集成文档](https://langfuse.com/resources/engineering/deepeval),把阶段 1–4 的指标包装成 experiment evaluator,分数与 reason 落到 Langfuse([Ragas 版集成](https://langfuse.com/integrations/frameworks/ragas)同理)。
2. 离线:Dataset + Experiment + CI 门禁(`langfuse/experiment-action`)([evaluation overview](https://langfuse.com/docs/evaluation/overview))。
3. 在线:给生产 trace 配 LLM-as-judge evaluator + 采样 rule;工具调用类检查用 code evaluator(免费全量)([llm-as-a-judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge))。
4. 闭环:生产坏案例 → dataset item → 回归测试;人工标注队列持续校准 judge([core concepts](https://langfuse.com/docs/evaluation/core-concepts))。
5. 可选扩展:promptfoo 做多模型对比矩阵与红队;TruLens/Phoenix 了解 OTel 系评估即可,不必与 Langfuse 重复建设;lm-evaluation-harness 跑一次 MMLU 子集,理解模型级评估与应用级评估的边界。

---

## 7. 参考来源

### 官方文档 / 仓库(全部于 2026-11-17 核实)

- DeepEval:[getting-started](https://deepeval.com/docs/getting-started) · [metrics-introduction](https://deepeval.com/docs/metrics-introduction) · [Faithfulness](https://deepeval.com/docs/metrics-faithfulness) · [G-Eval](https://deepeval.com/docs/metrics-llm-evals) · [trajectory-based evals](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals) · [GitHub](https://github.com/confident-ai/deepeval)
- Ragas:[RAG eval 教程](https://docs.ragas.io/en/stable/getstarted/rag_eval/) · [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) · [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) · [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) · [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) · [Agentic metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/) · [GitHub(新址)](https://github.com/vibrantlabsai/ragas)
- Inspect AI:[官方首页/Hello Inspect](https://inspect.aisi.org.uk/) · [Agents](https://inspect.aisi.org.uk/agents.html) · [Scorers](https://inspect.aisi.org.uk/scorers.html) · [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
- promptfoo:[getting-started](https://www.promptfoo.dev/docs/getting-started/) · [model-graded metrics](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/) · [RAG 评估指南](https://www.promptfoo.dev/docs/guides/evaluate-rag/) · [GitHub](https://github.com/promptfoo/promptfoo)
- TruLens:[quickstart notebook](https://github.com/truera/trulens/blob/main/examples/quickstart/quickstart.ipynb) · [RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) · [GitHub/README](https://github.com/truera/trulens)
- Phoenix:[evals quickstart](https://arize.com/docs/phoenix/evaluation/evals) · [GitHub](https://github.com/Arize-ai/phoenix) · [LICENSE(ELv2)](https://github.com/Arize-ai/phoenix/blob/main/LICENSE)
- lm-evaluation-harness:[GitHub/README](https://github.com/EleutherAI/lm-evaluation-harness)
- HELM:[GitHub/README(维护模式声明)](https://github.com/stanford-crfm/helm) · [维护模式政策](https://crfm-helm.readthedocs.io/en/latest/maintenance_mode/) · [文档](https://crfm-helm.readthedocs.io/)
- OpenAI Evals:[GitHub/README](https://github.com/openai/evals) · [提交历史](https://github.com/openai/evals/commits/main) · [平台化 Evals 产品文档](https://platform.openai.com/docs/guides/evals)
- Langfuse:[evaluation overview](https://langfuse.com/docs/evaluation/overview) · [core concepts](https://langfuse.com/docs/evaluation/core-concepts) · [LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) · [Score Analytics](https://langfuse.com/docs/evaluation/scores/score-analytics) · [AI agent evaluation 指南](https://langfuse.com/resources/engineering/ai-agent-evaluation) · [Ragas 集成](https://langfuse.com/integrations/frameworks/ragas) · [DeepEval 集成](https://langfuse.com/resources/engineering/deepeval) · [加入 ClickHouse 公告(2026-01-16)](https://langfuse.com/blog/joining-clickhouse) · [GitHub](https://github.com/langfuse/langfuse)
- ClickHouse:[收购 Langfuse 官方公告](https://clickhouse.com/blog/clickhouse-acquires-langfuse-open-source-llm-observability)
- LangSmith:[Evaluation 文档](https://docs.langchain.com/langsmith/evaluation)
- Agent 基准:[τ-bench](https://github.com/sierra-research/tau-bench) · [τ²/τ³-bench](https://github.com/sierra-research/tau2-bench) · [AgentBench](https://github.com/THUDM/AgentBench)

### 论文(一手)

- Ragas: Automated Evaluation of Retrieval Augmented Generation — [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)([HTML 全文](https://arxiv.org/html/2306.05685v4))
- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment — [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
- τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — [arXiv:2406.12045](https://arxiv.org/abs/2406.12045);τ²-bench — [arXiv:2506.07982](https://arxiv.org/abs/2506.07982)
- AgentBench: Evaluating LLMs as Agents — [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)
- GAIA: a benchmark for General AI Assistants — [arXiv:2311.12983](https://arxiv.org/abs/2311.12983)
- Holistic Evaluation of Language Models(HELM)— [TMLR / OpenReview](https://openreview.net/forum?id=iO4LZibEqW)
- BLEU: a Method for Automatic Evaluation of Machine Translation — [ACL P02-1040](https://aclanthology.org/P02-1040/)
- ROUGE: A Package for Automatic Evaluation of Summaries — [ACL W04-1013](https://aclanthology.org/W04-1013/)
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales — [DOI:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)
- TruLens Agent GPA(厂商自述,供参考)— [arXiv:2510.08847](https://arxiv.org/abs/2510.08847)
