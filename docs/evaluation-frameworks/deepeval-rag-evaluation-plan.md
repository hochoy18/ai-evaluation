# RAG 评估学习目标拆解(阶段 1 第 2 步 → 阶段 2)

> 对应总路线:`README.md` 第 6 节学习路径的阶段 1 第 2–3 步 + 阶段 2
> 姊妹文档:`deepeval-agent-evaluation-explained.md`(agent 线大纲,四步已走完)
> 本文定位:动手计划与验收标准,不是 API 参考。
> 打法复刻 agent 线的成功路径:**真实 app → 指标挂载 → 失败注入 → 理论对照**。
> 指标数据需求已于 2026-11-19 对 DeepEval v4.3.4 installed 源码核实。

---

## 总目标

**给一个真实 RAG 应用建立「能定位病灶」的评估体系,并理解同名指标在不同框架里的算法差异。**

最终产出四样东西:

| 产出 | 形态 |
|---|---|
| `src/deepeval_cases/rag_app.py` | 最小真实 RAG app(检索+生成,带插桩) |
| `src/deepeval_cases/test_rag.py` | pytest CI 门禁,4 指标全带 reason |
| 两端失败注入实验 | 三模式分数对照表(复刻 agent 线打法) |
| `docs/evaluation-frameworks/rag-eval-explained.md` | 体检报告 + DeepEval/Ragas 算法差异对照(带源码引用) |

---

## 前置知识(30 分钟,先画图再动手)

RAG 的失败面在**两端**,四个指标各管一摊:

```
用户问题 ──→ [检索端] ──→ retrieval_context ──→ [生成端] ──→ 答案
              │                                  │
     ContextualPrecision(查准:捞上来的有用吗)   Faithfulness(忠实:答案有据吗)
     ContextualRecall   (查全:该捞的捞到了吗)   AnswerRelevancy(切题:答的是问的吗)
```

**与 agent 线完全同构**:轨迹分说「病了」→ 组件分指「病灶」;RAG 里答案差 → 看四个分就知道是检索端还是生成端。

四个指标的数据需求(v4.3.4 源码 `_required_params` 核实):

| 指标 | input | actual_output | retrieval_context | expected_output | reference-free? |
|---|:-:|:-:|:-:|:-:|:-:|
| Faithfulness | ✓ | ✓ | ✓ | — | ✅ 可上线 |
| AnswerRelevancy | ✓ | ✓ | — | — | ✅ 可上线 |
| ContextualPrecision | ✓ | — | ✓ | ✓ | ❌ 仅离线 |
| ContextualRecall | ✓ | — | ✓ | ✓ | ❌ 仅离线 |

**关键认知**(README 4.5 节的落地):两个 context 指标要标准答案,只能离线用;生产在线评估(真实用户问题没有标准答案)几乎只能靠 Faithfulness/AnswerRelevancy。这就是 Ragas 三元组论文([arXiv:2309.15217](https://arxiv.org/abs/2309.15217))核心卖点是 reference-free 的原因。

**过关标准**:能不看表画出映射图,说清每个指标诊断哪端、为什么两个要 expected_output。

---

## 第 1 步:搭最小真实 RAG app(半天)

**做什么**:
- 知识库:5–8 篇短文档,继续医疗主题(发烧、咳嗽、高血压、糖尿病、挂号指南…),与已有 case 连贯;
- 检索器:先用**关键词打分**(零新依赖,十行代码),`@observe(type="retriever")` 插桩——agent 线学的 tracing 直接复用,检索 span 会出现在树里;
- 生成器:把检索结果塞进 prompt 让 LLM 作答,`@observe(type="llm")`;
- 顶层 `rag_app(question) -> (answer, retrieval_context)`。

**为什么不用向量库**:这一步的目标是理解评估,不是搭检索基建。关键词检索足够制造「检索端失败」;后面想升级 embedding 随时换,评估代码一行不改——**评估体系与被测系统解耦**,这本身就是个知识点。

**产出**:`rag_app.py`,可 import(跑题逻辑放 `if __name__ == "__main__":` 守卫,agent 线踩过的坑)。

**过关标准**:问「发烧四天干咳挂什么科」,能检索到相关文档并生成有据回答;`uv run deepeval inspect` 能看到 retriever span。

---

## 第 2 步:挂四指标 + 吃透算法(半天–1 天)

**做什么**:
1. 组装 `LLMTestCase(input, actual_output, retrieval_context, expected_output)`——**retrieval_context 是第一次真正用到的字段**,此前单轮 case 里它一直空着;
2. 写 `test_rag.py`(直接 pytest 形态,跳过脚本形态——已掌握,不重复学。**正典运行器 = `deepeval test run`**:本身即 pytest 包装器、产出 inspect 所需 test_run 快照,与「pytest 形态」不矛盾;裸 pytest 仅作快速调试,见架构 §9.3),4 指标 + `assert_test`;
3. **精读每个指标的 How Is It Calculated,手算一遍**:
   - **Faithfulness**:答案拆成 N 条 claims → 逐条对 retrieval_context 做 verdict → 分数 = 有据 claims / 总 claims(与 KnowledgeRetention 同属 verdict 型算法);
   - **ContextualPrecision**:对每个检索节点判「对得出标准答案有没有用」→ **按排名加权**(排第几很重要,这是它与 ContextualRelevancy 的本质区别);
   - **ContextualRecall**:标准答案拆句 → 逐句判「能否归因到检索内容」→ 比例;
   - **AnswerRelevancy**:答案拆 statements → 判与问题的相关性 → 比例。
4. 观察每个指标的 reason,验证手算与 judge 输出一致。

**产出**:`test_rag.py` 全绿,4 个 reason 都能读懂。

**过关标准**:给一条 faithfulness 的 reason,能指出它拆了哪几条 claim、哪条被判无据、分数怎么算出来的。

---

## 第 3 步:两端失败注入实验(半天,本线核心)

复刻 `agent_tools.py` 的三模式打法,但这次污染的是**两个不同的端**:

| 模式 | 注入点 | 预期分数变化 | 要验证的认知 |
|---|---|---|---|
| 基线 | 无 | 四指标都高 | — |
| `--bad-retrieval` | 检索器返回无关文档(问发烧给高血压文档) | Precision/Recall **暴跌**;Faithfulness **可能依然高** | **忠实 ≠ 正确**:答案可以完全忠实于捞上来的垃圾 |
| `--hallucinate` | 生成 prompt 诱导脱离上下文自由发挥 | Faithfulness **跌**;两个 context 指标**不动** | 检索没病,病在生成端 |

**为什么这是核心**:基线全绿说明不了任何事——**评估体系的价值在失败时才显现**。要亲眼看到「四个分数组成的指纹」如何区分两端故障,这就是体检报告的诊断能力。

特别关注 `--bad-retrieval` 下 Faithfulness 的表现:
- LLM 老老实实根据错误文档回答 → Faithfulness 高 + Context 指标低 = **检索端全责**;
- LLM 说「资料里没有相关信息」→ Faithfulness 也高(没瞎编)= 生成端尽责了。

两种都是好结果,但病灶都在检索端。

**产出**:三模式对照表 + 每格的 judge reason 摘录。

**过关标准**:盲测——给一组四个分数(不看代码),能说出哪端病了、为什么。

---

## 第 4 步:Ragas 对照(阶段 2 主体,1–2 天)

**做什么**:
1. **读论文**([arXiv:2309.15217](https://arxiv.org/abs/2309.15217))重点两节:为什么 reference-free、三维度如何对应检索/生成两端——与第 3 步实验现象互相印证;
2. `ragas>=0.4.4` **已在 pyproject 里**,不用装。把**同一批数据**(question/answer/contexts/ground_truth)喂给 Ragas 同名四指标;
3. **对比分数,读两边源码解释差异**——本步重头戏:
   - DeepEval:`.venv/lib/python3.12/site-packages/deepeval/metrics/faithfulness/`
   - Ragas:`.venv/lib/python3.12/site-packages/ragas/metrics/`
   - 预期发现:同名指标**算法不同**(claims 拆分粒度、verdict prompt、加权方式、是否用 NLI 蕴含…),分数差异不是 bug 而是设计差异;
4. 读 Ragas [Context Precision 文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/),理解变体(with/without reference、Non-LLM、ID-based)——同一个「查准」概念有多少种工程化;
5. 顺手体验 Ragas testset generation(没有测试集时自动造题)——解决以后「没有 golden 怎么办」的问题。

**产出**:`rag-eval-explained.md` 体检报告,含:
- 你的 RAG 系统两端健康度结论;
- **至少一对同名指标的算法差异对照表**(带两边源码文件/行号引用);
- 「什么场景选哪个框架的指标」的判断依据。

**过关标准**:能向别人解释「为什么 DeepEval 和 Ragas 的 Faithfulness 对同一条数据打出不同分」,并指出具体算法分歧点。

---

## 整体验收清单

- [ ] 能画 RAG 两端 × 四指标映射图,说清 reference-free/based 的线上/离线约束
- [ ] `test_rag.py` 进 CI,一条命令与单轮/多轮/agent 一起全量跑
- [ ] 三模式实验:看分数指纹能盲判病灶端
- [ ] 手算过 Faithfulness 的 claims 拆分
- [ ] 文档里至少一对同名指标算法差异,带源码引用
- [ ] (可选)Ragas 自动造题跑通一次

---

## 易错点预警

1. **retrieval_context 是 `list[str]`**,不是拼接好的字符串——每个元素是一个检索节点,ContextualPrecision 的排名加权按这个顺序;
2. **Faithfulness 也要 input**(源码核实),别只传 output + context;
3. Ragas 字段名与 DeepEval 不同(`user_input`/`response`/`retrieved_contexts`/`reference`),喂数据要做映射;
4. Ragas 默认 judge 是 OpenAI 官方模型名,要配成百炼 endpoint(与 DeepEval `OpenAIModel` 适配同理,`src/ai_evaluation/models.py` 经验可复用);
5. 两个框架的分数**不可直接比大小**,只能比「诊断结论是否一致」——第 4 步要建立的认知。

---

## 与已有知识的连接

| 已学(agent 线) | 本线对应 | 连接点 |
|---|---|---|
| 轨迹分 + 组件分双层定位 | context 指标 + faithfulness/answer relevancy | 同构诊断逻辑:拆维度是为了定位故障 |
| `@observe(type="llm"/"tool")` | `@observe(type="retriever")` | 同一套 tracing,新增 retriever span 类型 |
| `agent_tools.py` 三模式失败注入 | `--bad-retrieval` / `--hallucinate` | 同一打法:评估体系的价值在失败时显现 |
| verdict 型指标(KnowledgeRetention) | Faithfulness/ContextualRecall | 同族算法:原子检测 → 代码算比例 |
| `__main__` 守卫、pytest 形态 | rag_app.py / test_rag.py | 直接复用,不重复学 |
| judge temp 0.0、verbosity 扣分现象 | 四指标的 reason 观察 | 为阶段 3 judge 理论埋点 |

预计总耗时 3–4 天(与 README 阶段 2 估计一致)。
