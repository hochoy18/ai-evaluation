# DeepEval Agent 评测体系全景(数据流 · span 树 · 指标地图 · 实验发现)

> 定位:`deepeval-agent-evaluation-explained.md` 的 companion——那份讲「怎么跑」(五环节流水线),这份讲「**怎么连、怎么选、学到了什么**」。
> 姊妹文档:`eval-unit-three-levels.html`(动机讲义)、`deepeval-rag-evaluation-plan.md`(RAG 线,诊断逻辑同构)。
> 所有分数与 reason 均来自 `src/deepeval_cases/agent_tools.py` 实测(DeepEval v4.3.4,2026-11-17/19)。
> 代码:`agent_tools.py`(脚本形态三模式)、`test_agent.py`(pytest CI 门禁)。

---

## 1. 对象关系:体系的骨架

### 1.1 六个核心对象

| 对象 | 是什么 | 谁产生 |
|---|---|---|
| `Golden` | 只有题目的考卷(只有 input) | 你写 |
| `trace` | 一次运行的完整执行记录 | `evals_iterator` / pytest 插件开启收集 |
| `span` | 树上的一个节点(agent/llm/tool/retriever) | `@observe` 自动生成 |
| `LLMTestCase` | 组装好的用例(input/output/tools…) | DeepEval 从 trace **自动组装**,不用手写 |
| `metric` | 打分器,分轨迹级与组件级 | 你写,挂到 iterator 或 span |
| test run JSON | 结果落盘 | 每次 eval 后自动写 |

**与单轮 case 的本质区别**:单轮是你手工组装 `LLMTestCase`(题目+答案都准备好);agent 是 Golden 只给题,答案和过程由被测系统运行时真实产生,DeepEval 自动组装用例。

### 1.2 一次运行的完整数据流

```
Golden(只有题) ─→ evals_iterator ─→ 运行 agent
                                        │ @observe 自动记
                                        ▼
                                   trace(span 树)
                                        │ 自动组装
                                        ▼
                              LLMTestCase(input/output/tools…)
                                   ╱          ╲
                    轨迹指标读整棵树            组件指标读单个 span
                    (数据来自 update_current_trace)  (数据来自 update_current_span)
                                   ╲          ╱
                                    ▼
                          .deepeval/*.json ─→ deepeval inspect(TUI)
```

### 1.3 谁给谁供数据(体系的核心因果)

| 你写的代码 | 数据落在哪 | 谁读它 |
|---|---|---|
| `@observe(type=…)` | span 树 + 各节点 input/output/error | 所有轨迹指标 |
| `update_current_trace(input, output, tools_called, expected_tools)` | **testCase 顶层** `toolsCalled`/`expectedTools`(实测:不在 trace dict 里) | 轨迹级指标 |
| `update_current_span(tools_called, expected_tools, metrics)` | **该 span 自己的**字段 | 挂在该 span 的组件指标 |

两条关键因果:

1. **轨迹指标没有 trace 就是瞎子**——TaskCompletion/StepEfficiency 必须经 `evals_iterator` 或 pytest 插件运行,不能拿个普通 test_case 独立调用;
2. **组件指标只读自己 span 的数据**——所以工具决策数据必须用 `update_current_span` 喂到决策 span 上,喂到 trace 级它看不见。

### 1.4 两种运行形态,同一套数据流

| | 脚本形态 | pytest 形态 |
|---|---|---|
| 循环 | `evals_iterator` | `@pytest.mark.parametrize` |
| 断言 | 循环内自动 | `assert_test(golden=…, metrics=[…])` |
| trace 谁保持存活 | iterator 上下文 | **pytest 插件的 wrapper span** |
| 运行 | `uv run python xxx.py` | `uv run deepeval test run` |

源码核实(`evaluate/execute/trace_scope.py:63`):`assert_test(golden=…)` 从 `current_trace_context` 读活 trace,**必须 `deepeval test run`**——原生 pytest 会报 "No active trace found"。span 上动态挂的组件指标随 trace 一起自动评测,不用在 assert_test 里重复声明。

---

## 2. span 树解剖:执行树长什么样

### 2.1 基线实测树(agent_tools.py,正常路径)

```
travel_agent (agent) ─────────────────────── 树根
├── call_llm ① (llm)        决策: search_flights(纽约→洛杉矶, 2026-11-18)
│   └── chat qwen4-max-preview (llm) ← 自动生成
├── search_flights (tool)   output: [FL123 $450, FL456 $380]
├── call_llm ② (llm)        决策: book_flight(FL456)
│   └── chat qwen4-max-preview (llm)
├── book_flight (tool)      output: CONF-789
└── call_llm ③ (llm)        最终回复: 已为您预订…
    └── chat qwen4-max-preview (llm)
```

### 2.2 三个结构要点

**① tool span 的父亲是 agent,不是 llm。**
工具在 `travel_agent` 的循环里执行,不在 `call_llm` 里。LLM 只负责**决策**(返回 tool_calls),agent 负责**执行**(`TOOL_IMPLS[name](**args)`)。树忠实反映这个分工——所以组件指标(评判决策)挂在 `call_llm` 上,tool span 只记录执行事实(input/output/error)。

**② 每个 call_llm 里套着自动生成的 `chat {model}` span。**
用的是原生 `openai.OpenAI`,没改一行 import。机制:deepeval tracing 内置 OTel 自动插桩层(`tracing/otel/setup.py` 的 `_automatic_instrumentations`),trace 启动时自动 wrap openai 库,按 GenAI 语义约定生成 `chat {model}` span,记录原始请求/响应与 token 用量。每次 LLM 调用实际两层 span:手动外层(挂组件指标)+ 自动内层(API 细节)。

**③ 树的形状 = 循环的圈数。**
基线 3 圈 → 3 个 call_llm + 2 个 tool;失败注入/主动冗余各 4 圈 → 多一组 llm+tool。**轨迹指标数的「步」就是这棵树**。

### 2.3 失败时树记录什么

`--fail-search` 模式下,第一个 search_flights span:

```
error = '航班搜索 API 超时 (504)'
output = None
```

异常没有让程序崩溃:`@observe` 把错误记进 span → agent 的 `except` 把错误转成字符串回喂给 LLM → LLM 自主决定重试 → 多出一组 span。**错误变成了 messages 里的一条数据**——这就是 agent「自我修复」的本质,也是端到端评估永远看不到它的原因:最终答案照样对,只有 trace 记录了那次 504。

### 2.4 结果去哪了

| 文件 | 谁读 |
|---|---|
| `.deepeval/.latest_run_full.json` | `deepeval inspect` 默认读(滚动快照,每次覆盖) |
| `.deepeval/.latest_test_run.json` | 脚本提取 reason 用(内容几乎一样) |
| `data/test_run_*.json` | 设了 `DEEPEVAL_RESULTS_FOLDER` 时的单轮结果 |

inspect 解析优先级:命令行 PATH → `--folder`/`DEEPEVAL_RESULTS_FOLDER` → sqlite → 滚动快照。看历史运行直接传路径:`uv run deepeval inspect /path/to/snapshot.json`。

**落盘坑(2026-11-19 实测)**:裸 `@observe` 脚本运行**不写**本地快照——连包一层 `with trace():` 也不写(`.latest_run_full.json` 的 testCases 仍为 0)。快照只由 **eval 运行**产生(`evals_iterator` / `deepeval test run`)。所以 `rag_app.py` demo 跑完后 `inspect` 看到的还是上一次 eval 的旧数据;想看 retriever span,得等第 2 步的 test run。插桩收集与结果落盘是两件事:**trace 在内存里活着 ≠ 结果落了盘**。

---

## 3. 指标选择地图:从症状到药方

### 3.1 决策口诀

**结果对不对 → 端到端;过程好不好 → 轨迹;哪一步坏 → 组件。**

### 3.2 症状 → 指标 → 数据 → 挂载点

| 想诊断什么 | 指标 | scope | 必需数据 | 挂哪 |
|---|---|---|---|---|
| 任务到底办成没 | TaskCompletion | 轨迹 | trace + input/output | `evals_iterator` |
| 路径浪费吗(冗余步/绕路) | StepEfficiency | 轨迹 | trace | `evals_iterator` |
| 计划本身合理吗 | PlanQuality | 轨迹 | trace | `evals_iterator` |
| 执行偏离计划了吗 | PlanAdherence | 轨迹 | trace | `evals_iterator` |
| 这一步选对工具吗 | ToolCorrectness | 组件 | span 级 tools_called/expected_tools | 决策 llm span |
| 参数写对吗 | ArgumentCorrectness | 组件 | span 级 tools_called(input_parameters) | 决策 llm span |

### 3.3 打分机制:谁是代码,谁是 judge

| 指标 | 算法 | 说明 |
|---|---|---|
| ToolCorrectness(默认) | **确定性代码**:名字集合对比 | expected vs called 集合匹配,不经过 LLM;给 `available_tools` 才启用 LLM 评选择理由 |
| ArgumentCorrectness | LLM judge | 看 span input(上下文)+ 参数,判每个参数是否正确 |
| TaskCompletion / StepEfficiency | LLM judge | 读整棵树,推断任务意图,判完成度/每步必要性 |

**这解释了一个实验现象**(见 §4.3):wasteful 模式冗余 span 的 ToolCorrectness=0.0 是**代码算出来的**(expected=['book_flight'], called=['search_flights'],集合不匹配),不是 judge 的主观判断——组件定位因此是确定性的、可复现的。

### 3.4 expected_tools 怎么写:编码「最小正确路径」

组件分的定位能力,本质来自你给每个决策点写的期望。agent_tools.py 的按阶段推断:

```python
searched = any(m["role"] == "tool" and "price" in m["content"] for m in messages)
expected = [ToolCall(name="book_flight")] if searched else [ToolCall(name="search_flights")]
```

没搜过 → 该搜;搜过了 → 该订。冗余搜索之所以被定位,就是因为它违反了这条编码好的最小路径。

---

## 4. 实验发现归档:分数教会我们的事

### 4.1 三模式分数对照(实测)

| 指标 | 基线 | `--fail-search`(被迫重试) | `--wasteful`(主动冗余) |
|---|---|---|---|
| TaskCompletion(轨迹) | 0.98 | 0.95 | **0.95** ← 答案照样对 |
| StepEfficiency(轨迹) | 0.75 | 0.75 ← 没降! | **0.50 ❌** ← 过程烂 |
| ToolCorrectness(组件) | 1.0 ×2 span | 1.0 ×3 span | **0.67**(1 个 span 挂 0 分) |
| ArgumentCorrectness(组件) | 1.0 ×2 | 1.0 ×3 | 1.0 ×3 |

### 4.2 发现一:judge 区分「被迫重试」与「主动浪费」

原预测失败注入会拉低 StepEfficiency,实测没降。judge 的 reason:

> fail-search:"the duplicate search… was **forced by a 504 API timeout, not speculative**"
> wasteful:"second search_flights call with **byte-identical parameters** purely to re-verify… **4 LLM calls where 3 sufficed**"

**StepEfficiency 不是数步数,是判每一步的必要性。** 错误恢复是合理步骤,不扣分;无错误的重复才是冗余。要制造「答案对,过程烂」,得注入**主动**浪费,不是被动故障。

### 4.3 发现二:选错工具 ≠ 写错参数

wasteful 模式三个决策 span 逐个打分:

```
llm span① 决策=search_flights  → ToolCorrectness 1.0 ✅  ArgumentCorrectness 1.0 ✅
llm span② 决策=search_flights  → ToolCorrectness 0.0 ❌  ArgumentCorrectness 1.0 ✅ ← 病灶
          reason: expected ['book_flight'], called ['search_flights']
llm span③ 决策=book_flight     → ToolCorrectness 1.0 ✅  ArgumentCorrectness 1.0 ✅
```

冗余搜索的**参数写得完全正确**(城市、日期都对),错的是「不该再搜一次」这个**选择**。ToolCorrectness 管选择,ArgumentCorrectness 管参数,互不替代。轨迹分说「病了」(SE 0.50),组件分说「病在第②步」——与 RAG 拆 faithfulness/context 定位检索端 vs 生成端完全同构。

### 4.4 发现三:高效 ≠ 完成(两个轨迹分必须一起读)

基线第一次跑:system prompt 没给当前日期,LLM 解析不了「明天」,反问用户而不调工具:

- TaskCompletion **0.05**(任务根本没办)
- StepEfficiency **1.00**(只有一步,零冗余)

**高效地什么都没干。** 单看任何一个轨迹分都会被骗。修复:system prompt 注入当前日期(真实 agent 的标准做法)。

### 4.5 发现四:「答案对,过程烂」只有轨迹评估看得见

wasteful 模式最终回复完全正确(订到最便宜 FL456,确认号齐全),TaskCompletion 0.95。端到端评估(只看 input/output)对那次冗余搜索**完全失明**。这是评估单位必须从文本升级到执行树的根本原因——**评估单位必须匹配系统的失败面**。

---

## 5. judge 怎么打分(阶段 3 埋点)

轨迹 judge 的工作流:**读整棵 span 树 → 从 input 推断任务意图 → 对照执行链判完成度/必要性 → 给分 + reason**。

已观察到的 judge 行为特征:

- 判**必要性**而非步数(§4.2);
- 会扣**风格分**:最终回复的冗余比较、emoji、客套话被 StepEfficiency 视为 excess(基线 SE 0.75 而非 1.0 的原因)——这是 verbosity bias 的温和表现;
- 分数有波动:app temp 0.7 时同一 case 跑出过 1.0/0.9/0.6;app 与 judge 都设 0.0 后决策可复现(组件级的确定性代码打分则完全无波动)。

这些现象在阶段 3(judge 理论)会系统展开:MT-Bench 三大偏差(arXiv:2306.05685)、G-Eval 概率加权(arXiv:2303.16634)、人工校准 Cohen's Kappa。

---

## 6. 一页总结

```
评估单位:文本快照 → 对话 → 执行树(失败面在哪,单位就到哪)
对象链:  Golden → trace(span 树) → 自动组装 LLMTestCase → 指标 → JSON → inspect
数据通道:update_current_trace 喂轨迹级;update_current_span 喂组件级
选指标:  结果对不对→端到端;过程好不好→轨迹;哪一步坏→组件
打分机制:ToolCorrectness 默认是代码集合对比;其余靠 LLM judge 读树
实验铁律:基线全绿说明不了什么,评估体系的价值在失败时才显现
两个一起读:TaskCompletion × StepEfficiency(高效≠完成)
两个不互替:ToolCorrectness(选择) × ArgumentCorrectness(参数)
```
