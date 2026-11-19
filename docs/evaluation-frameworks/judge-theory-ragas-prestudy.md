# Judge 理论 × Ragas 对照预研(阶段 2/3)

> 本文是 RAG 评估学习路径**阶段 2(Ragas 对照)+ 阶段 3(judge 理论)**的文献预研,配套动手计划 [deepeval-rag-evaluation-plan.md](./deepeval-rag-evaluation-plan.md)。
> 研究日期:2026-11-19。纯文献与源码研究,未运行评测、未调用 LLM API。
> 源码版本:**DeepEval v4.3.4(installed)**、**Ragas 0.4.4(installed**,`.venv` 内 `ragas-0.4.4.dist-info` 核实;两包均已安装,全部源码引用为一手本地核实)。
> G-Eval 框架与 DeepEval GEval 实现已有专文,本文**交叉引用不重复**:[geval-framework-explained.md](./geval-framework-explained.md)、[deepeval-geval-explained.md](./deepeval-geval-explained.md);百炼适配见 [deepeval-bailian-adapter.md](./deepeval-bailian-adapter.md)。

**源码引用缩写**(下文 `DE:` / `RG:` 均相对仓库根目录):

- `DE:` = `.venv/lib/python3.12/site-packages/deepeval/`
- `RG:` = `.venv/lib/python3.12/site-packages/ragas/`

---

## 1. Judge 偏差分类学(MT-Bench 论文)

来源:*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*(Zheng et al., NeurIPS 2023,[arXiv:2306.05685](https://arxiv.org/abs/2306.05685),以下节号按 arXiv v4 HTML 版)。

### 1.1 三种 judge 形态(§3.1)

| 形态 | 做法 | 弱点 |
|---|---|---|
| Pairwise comparison | 给问题 + 两个答案,判谁更好/平局 | 选手数增长时配对数平方增长,不可扩展 |
| Single answer grading | 直接给单个答案打分 | 难分辨细微差异;**绝对分比相对 pairwise 结果更不稳定**(换 judge 模型时波动更大)——本项目 StepEfficiency/GEval 都属此类 |
| Reference-guided grading | 提供参考答案再打分 | 需要参考解 |

### 1.2 四大局限(§3.3)

| 偏差 | 定义 | 实验证据 |
|---|---|---|
| **Position bias** | 偏好特定位置的答案 | 用 GPT-3.5 temp 0.7 对同一问题生成两个相似答案,交换位置测一致性(Table 2):GPT-4 仅 **65.0%** 一致,GPT-3.5 46.2%,Claude-v1 23.8%;多数 judge 偏向**第一个**位置;rename 实验证明 Claude 还有名字偏差(偏 "Assistant A") |
| **Verbosity bias** | 偏好更长更啰嗦的回答,即使更不清晰、不准确 | "repetitive list" 攻击:把含编号列表的 23 个答案用 GPT-4 改写列表(**不加新信息**)插到原列表前。judge 若认为新答案更好即攻击成功(Table 3):Claude-v1 与 GPT-3.5 失败率均 **91.3%**,GPT-4 **8.7%**。对照:完全相同的答案 judge 能正确判平局 |
| **Self-enhancement bias** | 偏好自己生成的答案 | GPT-4 评自己胜率高出人类评价约 **10%**,Claude-v1 约 **25%**;但 GPT-3.5 不偏自己,且无法在「不改质量的前提下把答案改写成另一模型风格」做受控实验——**论文自认数据不足以统计断定此偏差存在**(引用时注意这个限定) |
| **数学/推理判分能力有限** | 判不对数学/推理题 | 值得注意:GPT-4 **单独能解对**的初等数学题,判分时却被给出的错误答案带偏(Appendix Fig 12/13);GPT-3.5、Claude 同样 |

### 1.3 缓解手段(§3.4)

| 手段 | 针对 | 效果与代价 |
|---|---|---|
| **交换位置**(conservative) | position bias | 调两次 judge,两个顺序都赢才算赢,不一致判平局;激进版是随机分配位置 |
| **Few-shot judge** | position bias | 3 个好判例(A 胜/B 胜/平局)使 GPT-4 一致性 **65.0% → 77.5%**(Table 13);但 prompt 变长 API 成本 **4×**,且可能引入新偏差,一致性高≠准确率高 |
| **CoT judge** | 推理判分 | 让 judge 先独立解题再判分;失败率 14/20 → 6/20(Table 4),但 judge 解题时常犯与被评答案**相同的错误**(仍被上下文带偏) |
| **Reference-guided judge** | 推理判分 | 先**独立生成** judge 自己的答案,再作为 reference 放进判分 prompt;失败率 **14/20(70%)→ 3/20(15%)**(Table 4)。关键在「独立生成」,避免被污染 |
| 微调 judge 模型 | 综合 | Vicuna-13B 在 arena 数据上微调,初步结果见 Appendix F |

### 1.4 judge 与人工一致性的度量(§4.1–4.2,Appendix D.3)

- **Agreement 定义**:从两类 judge 中各随机选一个**不同**个体,在随机问题上判断一致的概率(不是简单逐条对比,而是「随机配对」口径)。
- 两种口径:**S1** 含平局、把位置交换不一致计为平局;**S2** 只计非平局票。
- 核心结果(§4.2,Table 5):GPT-4 与人类专家 **S2 一致性 85%**,高于**人类之间 81%**——「LLM judge 达到人类间一致性水平」这一著名论断的出处。
- 分歧案例中,人类认为 GPT-4 判断合理的占 **75%**,愿意改自己选择的占 **34%**(§4.2)。
- 注意:这是**偏好判断(pairwise/grading)的一致性**,不是本项目回归式指标分数的校准;后者要用 §3 的 Kappa/TPR-TNR 方法。

---

## 2. G-Eval 打分机制:论文层细节与工程对照

框架三组件(prompt + auto-CoT + form-filling 概率加权)与 DeepEval 参数映射**已覆盖于** [geval-framework-explained.md](./geval-framework-explained.md) 与 [deepeval-geval-explained.md](./deepeval-geval-explained.md),此处只补论文原文细节与实现差异。来源:[arXiv:2303.16634](https://arxiv.org/abs/2303.16634)(v3,EMNLP 2023)。

### 2.1 概率加权为什么必要(§2 Scoring Function)

直接让 LLM 填表打分有两个问题(论文原文列举):

1. **某个数字主导分布**(如 1–5 分制里的 3)→ 分数方差低 → 与人类判断相关性低;
2. **LLM 即使被要求输出小数也只输出整数** → 大量平分(ties),无法区分文本间的细微差异。

解法:对 prompt 预定义的分数集合 S = {s₁,…,sₙ},取各分数 token 的概率并**在 S 上归一化**,加权求和:

```
score = Σᵢ p(sᵢ) · sᵢ
```

### 2.2 论文实现 vs DeepEval 实现(关键工程差异)

| 维度 | 论文(§3.1 实验设置) | DeepEval v4.3.4 |
|---|---|---|
| 概率来源 | GPT-3.5(text-davinci-003)temp 0 取 logprobs;**GPT-4 当时不支持 token 概率输出,用 `n=20` 采样 20 次估计概率分布** | 直接读 `top_logprobs=20`(一次调用的 logprobs),见 [deepeval-geval-explained.md](./deepeval-geval-explained.md) |
| 分数域 | 1–5 | 默认 0–10,归一化到 0–1 |
| 无概率时 | 论文未讨论回退 | 回退原始整数分(粒度 0, 0.1, …, 1.0)——百炼兼容接口未必返回 logprobs,正是此回退路径 |

即:论文的「20」是**采样次数**,DeepEval 的「20」是**logprobs 候选数**,机制不同,目标一致(估计分数分布)。

### 2.3 消融与局限(§3.4,§4 Analysis)

- SummEval 摘要任务平均 **Spearman 0.514**,超过此前所有自动指标(§3.4,Table 1)。
- **CoT 消融**(§4 "The Effect of Chain-of-Thoughts"):带 CoT 的 G-Eval-4 在所有维度相关性更高,fluency 尤甚。
- **概率加权消融**(§4 "The Effect of Probability Normalization"):加权提升 **Spearman** 但降低 **Kendall-Tau**——因为整数打分产生大量 ties,而 Kendall-Tau 不计 ties,虚高。解读:加权买到的是「细粒度排序能力」,不是所有相关系数都涨。
- **偏袒 LLM 生成文本**(§4 "Will G-Eval prefer LLM-based outputs?"):即使人类评委更偏好人写摘要,G-Eval-4 **始终给 GPT-3.5 摘要更高分**。论文给出两个假说:① 高质量摘要本身难评(该数据集人类标注者间 Krippendorff α 仅 0.07);② 模型生成与评估时**共享同一套「好文本」概念**。警告:若把这种 judge 分数当奖励信号自我调优,会**自我强化**(over-fit 到自己的评估标准)。
- 与 MT-Bench self-enhancement 的关系:两者同属「自偏好」家族,但 G-Eval 的证据是**分数系统性偏高**(grading),MT-Bench 的是**胜率偏高**(pairwise)且自认统计上不能定论。

---

## 3. 校准实践:Kappa、校准集与 judge prompt 迭代

### 3.1 Cohen's Kappa:定义与解读

**一手文献**:Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. DOI: 10.1177/001316446002000104。
**权威统计参考**(开放获取,公式与解读已逐字核实):McHugh, M.L. (2012). *Interrater reliability: the kappa statistic*. Biochemia Medica 22(3), 276–282. [PMC3900052](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/)。

- **动机**(Cohen 1960,转述自 McHugh §Introduction):简单百分比一致率(percent agreement)**无法扣除碰巧一致**——评分者不确定时会猜,猜测也会撞合。
- **公式**(McHugh §Calculation of Cohen's kappa):

```
κ = (Pr(a) − Pr(e)) / (1 − Pr(e))
```

Pr(a) = 实际观察到的一致率;Pr(e) = 由两个评分者的**边际分布**估计的随机一致率。κ ∈ [−1, 1],0 = 仅随机水平的一致,1 = 完全一致。

- **解读区间**(Landis & Koch 1977, *Biometrics* 33(1):159–174, DOI: 10.2307/2529310;原文付费墙未直读,区间经 McHugh 2012 与 Langfuse 官方文档两处转载核实):

| κ | 解读 |
|---|---|
| ≤ 0 | 无一致(差于随机) |
| 0.01–0.20 | none to slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| 0.61–0.80 | substantial |
| 0.81–1.00 | almost perfect |

- **两个警告**:① McHugh 认为上述区间对高风险场景**过宽**,主张 κ < 0.60 即「一致不足,结果不可信」;② κ 的机遇校正假设(评分者按边际比例独立猜测)本身有争议(McHugh §Discussion)。**类别不平衡时裸一致率会骗人**:错误率 5% 时全判 Pass 也有 95% 一致率——应分开报 **TPR/TNR**(Hamel 指南,见 §3.2;Langfuse Score Analytics 的 advanced 模式同理)。

### 3.2 30–50 条校准集的构建流程

实践一手来源:Hamel Husain, *Creating a LLM-as-a-Judge That Drives Business Results*([hamel.dev/blog/posts/llm-judge](https://hamel.dev/blog/posts/llm-judge/),下称 Hamel);平台实践:Langfuse 官方校准指南(见 §3.3)。

1. **采样真实数据**:从生产 trace 取代表性样本,覆盖各种结局与已知失败案例(Hamel Step 2;Langfuse skill `judge-calibration.md` §1 同款要求)。
2. **领域专家做二元 Pass/Fail + 详细 critique**(Hamel Step 3):起步阶段**不要 1–5 分制**(「一堆 1–5 分通常是坏评估流程的标志」);critique 要详细到「新员工能看懂」,因为它直接进 judge 的 few-shot prompt。
3. **样本量**(Hamel "How many examples do you need?" 与 train/dev/test 节):
   - 发现失败模式、写 rubric:**从 ~30 条开始**,直到不再出现新失败模式;
   - **验证 judge** 需要更大标注集:每个失败模式 ~100 条,< 60 条置信区间太宽;
   - 划分:train 10–20%(做 few-shot),dev/test 各 40–45%,**每个集合每种标签尽量 30–50 条**;dev/test 样本**绝不进 judge prompt**。
4. **度量**:以人工标签为 ground truth,报 judge 的 **TPR 与 TNR**(而非裸 agreement);类别/布尔分数再算 **Cohen's Kappa + 混淆矩阵**(Langfuse Score Analytics)。
5. **迭代 judge prompt**(Hamel Step 5):把专家 critique 作为 few-shot 例子 → 跑 judge → 电子表格逐条对照人工标签与 judge 输出 → 修 prompt → 重跑,跟踪一致率收敛。Hamel 在 Honeycomb 案例 **3 轮迭代达到 >90% 一致**。理论支撑:MT-Bench §3.4 few-shot judge 显著提升一致性(§1.3)。
6. **在 unseen 数据上测错误率**,防止对校准集过拟合(Hamel Step 6)。

### 3.3 Langfuse 平台校准流程(补充)

来源:[Calibrate LLM-as-a-judge 指南](https://langfuse.com/guides/llm-as-a-judge-calibration-skill)、[Score Analytics 文档](https://langfuse.com/docs/evaluation/scores/score-analytics)、本地 langfuse skill `references/judge-calibration.md`(51 行,已读)。

- 流程:标注数据集(`judge-calibration/<judge-name>`)+ Prompt Management 存候选 prompt → Experiments SDK 对每条数据跑 judge → Boolean evaluator 比对 expected output → 报告 → **逐条分析分歧** → 改 prompt(需用户批准)→ 重跑 → 批准后打 `production` 标签部署。
- 质量门槛(skill §3):**85% 总体准确率视为「相当好≈人类判断水平」,但必须每个类别都可接受**;advanced 模式报混淆矩阵 + precision/recall/F1/TPR/TNR(指南示例:35 条数据,TPR/TNR 均过 0.90 门槛)。
- 分歧分析实例(指南):一条「用户拒绝 app 内转人工但声称要直接邮件 CEO」的边界案例同时触发 prompt 两条规则——**修法是明确规则优先级,收紧 prompt**,而非换模型。
- 上线后持续校准:annotation queue 人工抽样打分,Score Analytics 对比 judge 分与人工分(κ + 混淆矩阵;文档示例 κ=0.85 → "Almost Perfect")。

---

## 4. Ragas 论文核心(arXiv:2309.15217)

来源:*Ragas: Automated Evaluation of Retrieval Augmented Generation*(Es et al., EACL 2024 demo,[arXiv:2309.15217](https://arxiv.org/abs/2309.15217),v1)。

### 4.1 为什么 reference-free(§1,§3)

构建 RAG 系统时通常**没有人工标注的参考答案**,因此论文聚焦完全自包含、无需 ground truth 的指标,以加快评估迭代。三个质量维度(§3):

| 维度 | 定义 | 诊断哪端 |
|---|---|---|
| **Faithfulness** | 答案必须 grounded 于给定上下文,避免幻觉,上下文可作答案的证成 | 生成端(是否忠于检索结果) |
| **Answer Relevance** | 答案要回应实际问题;**不考察事实性**,惩罚不完整与冗余信息 | 生成端(是否切题) |
| **Context Relevance** | 检索上下文要聚焦、尽量少无关信息——长上下文既贵,且 LLM 对中段信息利用差(引 Liu et al. 2023 "lost in the middle") | 检索端 |

这正是 [deepeval-rag-evaluation-plan.md](./deepeval-rag-evaluation-plan.md) 前置知识里「两个 context 指标要 expected_output 只能离线,线上靠 faithfulness/answer relevancy」认知的论文出处。

### 4.2 三个指标的原始算法(§3)

- **Faithfulness**:LLM 把答案拆成 statements 集合 S → 逐条用 NLI 式 prompt 验证「能否由上下文推出」(Yes/No + 解释)→ 支持集 V,**F = |V| / |S|**。
- **Answer Relevance**:由答案**反向生成 n 个问题** qᵢ → 与原问题 q 做 embedding 余弦,**AR = (1/n) Σ cos(q, qᵢ)**。
- **Context Relevance**:LLM 从上下文**抽取对回答必要的句子**(不许改句;抽不出则返回 "Insufficient Information"),**CR = 抽取句数 / 上下文总句数**。

### 4.3 与人工判断的对齐验证(§4 WikiEval,§5 Table 1)

- WikiEval:50 个 2022 年后事件的 Wikipedia 页面(刻意超出模型训练截止),两名标注者,faithfulness/context relevance 标注一致率 ~95%,answer relevance ~90%。
- 成对比较准确率(Table 1):**Ragas 0.95 / 0.78 / 0.70**(faith / AR / CR)vs 直接让 ChatGPT 打 0–10 分的 GPT Score **0.72 / 0.52 / 0.63** vs GPT Ranking **0.54 / 0.40 / 0.52**。
- 结论:**「原子拆解 + 代码聚合」显著优于「LLM 直接打总分」**——这是 QAG 范式(DeepEval 四指标同族)有效性的论文级证据。CR 最难:ChatGPT 在长上下文里挑关键句子经常失手。
- 注意演化:论文的 context relevance(句子比例)**不是**今天 Ragas 的 context_precision(排名加权 AP);AR 的反向生成问题 + 余弦则一脉相承至今(§5 源码核实)。

---

## 5. 四指标算法对照表(重头戏)

两边均为 installed 源码一手核实。Ragas 以 **collections API(0.4 现行推荐)**为准;legacy `ragas.metrics._*` 模块算法相同(如 `RG:metrics/_faithfulness.py:146-182` 同为 statement generator + NLI),官方文档注明 legacy API 0.4 弃用、1.0 移除([docs.ragas.io context_precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) "Deprecation Timeline")。

### 5.1 Faithfulness —— 分歧最大的一对

| 维度 | DeepEval v4.3.4 | Ragas 0.4.4(collections) |
|---|---|---|
| 上下文处理 | **先从 retrieval_context 抽 truths**(LLM 调用 1),verdict 对着 truths 判而非原文 | 直接 `"\n".join(retrieved_contexts)` 原文(`RG:metrics/collections/faithfulness/metric.py:124`) |
| 答案拆分 | claims:「factual, undisputed truths」,要求带完整上下文、不摘樱桃、不用先验知识(`DE:metrics/faithfulness/templates/generate_claims.txt:1,19-21`) | statements:**给定 question + answer**,逐句拆成「完全可独立理解、无代词」的原子陈述(`RG:metrics/collections/faithfulness/util.py:35`) |
| 判定范式 | verdict ∈ yes/no/borderline;**no 仅当上下文直接矛盾**(DIRECTLY CONTRADICTS),**无据支持 = borderline**(`DE:templates/metrics/templates.json:288`) | NLI 二值 0/1:「能否由上下文**直接推出**」,推不出(含无信息)= 0(`RG:.../faithfulness/util.py:86`,示例 :106-120) |
| 打分公式 | 默认 passing = (YES, BORDERLINE) → **score = 1 − 矛盾率**;`penalize_ambiguous_claims=True` 才只算 YES(`DE:metrics/faithfulness/faithfulness.py:408-414`) | **score = 支持数 / 总数**(`RG:.../faithfulness/metric.py:148-163`) |
| 空输入 | 无 claims → **1.0**(`DE:metrics/utils/qag.py:118,129-130` empty_score=1) | 无 statements → **NaN**(`RG:.../faithfulness/metric.py:119-121`) |
| LLM 调用 | truths + claims + verdicts + reason = 4 次(`DE:.../faithfulness.py:141-147`) | statements + NLI = 2 次,reason 内嵌于 NLI 输出(`RG:.../faithfulness/metric.py:116-125`) |

**核心结论**:默认配置下两者测的**不是同一个量**——DeepEval 测「矛盾率」(无据但无害的 claim 被 borderline 放行),Ragas 测「支持率」(无据即扣分)。同一条含「上下文未提及但不矛盾」陈述的答案,**Ragas 分更低是设计使然,不是 bug**。要对齐语义,DeepEval 需开 `penalize_ambiguous_claims=True`。另注意 DeepEval 对 truths 的二次抽取可能丢失上下文信息。

### 5.2 Answer Relevancy —— 范式完全不同

| 维度 | DeepEval v4.3.4 | Ragas 0.4.4 |
|---|---|---|
| 算法 | 答案拆 statements(含糊句、孤立词也算,`DE:metrics/answer_relevancy/templates/generate_statements.txt:1`)→ 逐条判与 input 的相关性 yes/no/borderline(`.../templates/generate_verdicts.txt:1-3,29-34`) | **反向生成问题**:由 response 生成 strictness=3 个问题 + noncommittal 标志(`RG:metrics/collections/answer_relevancy/metric.py:112-124`,`util.py:34-36`) |
| 打分公式 | score = (YES+BORDERLINE)/总数(`DE:metrics/answer_relevancy/answer_relevancy.py:326-331`);borderline = 「可能是支撑信息」也放行 | **score = mean cos(原问题, 生成问题) × (全员 noncommittal ? 0 : 1)**(`RG:.../answer_relevancy/metric.py:142-154`) |
| 依赖 | 纯 LLM verdict,**无需 embeddings** | LLM + **embeddings**(余弦相似度) |
| 特色惩罚 | 无关 statement 拉低比例 | 敷衍回答("I don't know" 类)直接**清零**;冗余信息稀释余弦(论文 §3 AR 定义) |
| 空输入 | 无 statements → 1.0(qag empty_score=1) | 无生成问题 → 0.0(`RG:.../metric.py:126-127`) |

**核心结论**:verdict 比例 vs 嵌入余弦,分数**不可直接比大小**;Ragas 版对「答非所问但句句相关」不敏感(问题相似即可),DeepEval 版对「句句相关但整体回避问题」不敏感(无 noncommittal 门)。

### 5.3 Contextual Precision —— 同核不同变体

| 维度 | DeepEval v4.3.4 | Ragas 0.4.4 |
|---|---|---|
| 逐节点 verdict | 「对得出 expected output 是否 **remotely useful**」,yes/no + reason 需引用原文(`DE:metrics/contextual_precision/templates/generate_verdicts.txt:1,4`) | 「对得出 given answer 是否 useful」,0/1 + reason(`RG:metrics/collections/context_precision/util.py:31`);**每个 context 单独一次 LLM 调用**(`RG:.../context_precision/metric.py:103-112`) |
| 排名加权公式 | **AP = Σₖ precision@k · relₖ / 相关节点总数**;无 verdict → 0,无相关节点 → 0(`DE:metrics/contextual_precision/contextual_precision.py:366-389`) | **同构 AP**,分母 = cumsum + **1e-10**(无相关节点时 ≈0 而非硬 0)(`RG:.../context_precision/metric.py:118-138`) |
| reference 变体 | 只有 with-reference(必需 expected_output,`DE:.../contextual_precision.py:60-63`);无 reference 的「查准」要用另一指标 ContextualRelevancy(不做排名加权) | **WithReference**(对比 reference)/ **WithoutReference = ContextUtilization**(对比 response,可线上)(`RG:.../context_precision/metric.py:21,141,261,297`) |
| Non-LLM 变体 | 无 | `NonLLMContextPrecisionWithReference`:每个 retrieved context 与 reference_contexts 逐一算**字符串相似度**(默认 Levenshtein,`RG:metrics/_string.py:62-81`),max ≥ 0.5 → 二值化 → 同一 AP(`RG:metrics/_context_precision.py:187-247`) |
| ID-based 变体 | 无 | `IDBasedContextPrecision`:**命中 ID 数 / 检索 ID 总数**,纯集合运算,**无排名加权**(`RG:metrics/_context_precision.py:251-305`;公式见官方 docs) |

**核心结论**:LLM 版两边**排名加权核心同构**(相关节点排越前分越高,官方 docs 示例:无关 chunk 排第 2 不影响分、排第 1 则 0.5);差异在 ① DeepEval 措辞 "remotely useful" 更宽松;② Ragas 变体谱系完整(with/without reference × LLM/Non-LLM/ID-based),把「查准」从语义判断一路退化到零成本集合对比——**同一概念的多档工程化,精度换成本**。

### 5.4 Contextual Recall —— 最接近的一对

| 维度 | DeepEval v4.3.4 | Ragas 0.4.4 |
|---|---|---|
| 算法 | expected_output 逐**句**判「能否归因到检索节点」,单次 LLM 调用出全部 verdicts(yes/no),reason 需指明第几节点并引原文(`DE:metrics/contextual_recall/templates/generate_verdicts.txt:1-3`) | reference 逐 **statement** 判 attributed 0/1 + reason,同样单次 LLM 调用(`RG:metrics/collections/context_recall/metric.py:109-127`,`util.py:50-52`) |
| 打分公式 | score = YES/总数,passing 仅 YES(`DE:metrics/contextual_recall/contextual_recall.py:262-268`) | score = attributed/总数(`RG:.../context_recall/metric.py:126-127`) |
| 空输入 | 无 verdict → **0.0**(empty_score=0,`DE:.../contextual_recall.py:267`) | 无 classifications → **NaN**(`RG:.../context_recall/metric.py:122-123`) |
| 拆分粒度 | 模板措辞为 sentence(system_one 路径用 `split_sentences`,`DE:.../contextual_recall.py:344`) | LLM 自行拆 statement(粒度由模型定) |

**核心结论**:算法同构(归因比例),分数差异主要来自**拆分粒度**(句 vs 原子陈述)与空值约定(0 vs NaN)。这是验证「同数据喂两框架」时最该先对齐的一对。

### 5.5 汇总:同名 ≠ 同算法

| 指标 | 范式分歧 | 默认空值 | 谁更严 |
|---|---|---|---|
| Faithfulness | 矛盾检测(borderline 放行)vs NLI 支持检测 | DE 1 / RG NaN | **Ragas**(无据即扣) |
| Answer Relevancy | verdict 比例 vs 反向问题余弦 + noncommittal 门 | DE 1 / RG 0 | 视数据而定 |
| Context Precision | 同构 AP;Ragas 多 4 变体 | 均 ≈0 | 措辞 DE 略宽 |
| Context Recall | 同构归因比例;句 vs statement | DE 0 / RG NaN | 近似 |

---

## 6. Ragas 工程接口与百炼适配

### 6.1 字段名映射

| Ragas(`SingleTurnSample`,`RG:dataset_schema.py:54-93`) | DeepEval(`LLMTestCase`) | 备注 |
|---|---|---|
| `user_input` | `input` | |
| `response` | `actual_output` | |
| `retrieved_contexts: list[str]` | `retrieval_context: list[str]` | 都是节点列表,顺序即排名 |
| `reference` | `expected_output` | |
| `reference_contexts` / `reference_context_ids` | (无对应) | Non-LLM / ID-based 变体专用 |

### 6.2 两种用法

```python
# A. collections 单指标直评(0.4 推荐,无需 dataset)
from ragas.metrics.collections import Faithfulness
result = await Faithfulness(llm=llm).ascore(
    user_input=q, response=a, retrieved_contexts=ctxs)   # RG:metrics/collections/faithfulness/metric.py:88-90

# B. 批量 evaluate(legacy 指标 + EvaluationDataset)
from ragas import EvaluationDataset, SingleTurnSample, evaluate
ds = EvaluationDataset.from_list([{"user_input": q, "response": a,
    "retrieved_contexts": ctxs, "reference": ref}, ...])  # RG:dataset_schema.py:318,391
evaluate(dataset=ds, metrics=[...], llm=llm, embeddings=emb)  # RG:evaluation.py:349
```

collections 指标**只接受 modern instructor LLM,legacy wrapper 会被明确拒绝**(各类 docstring,如 `RG:metrics/collections/faithfulness/metric.py:34-35`);legacy `MetricWithLLM` 两种都收(`RG:metrics/base.py:161-162`)。

### 6.3 judge LLM 配成百炼 OpenAI 兼容端点

**不需要 langchain wrapper**。现行路径是 `llm_factory` + OpenAI SDK client(instructor 结构化输出):

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
client = AsyncOpenAI(
    api_key="sk-百炼key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 或本项目 token-plan endpoint,见 adapter 文档
)
llm = llm_factory("qwen-max", client=client, temperature=0)  # RG:llms/base.py:624-632
```

- `llm_factory(model, provider="openai", client=..., mode=..., **kwargs)`:默认 instructor `Mode.JSON`;**后端不支持 `response_format` 参数时用 `Mode.MD_JSON`**(`RG:llms/base.py:658-663`);docstring 自带 custom `base_url` 示例(`RG:llms/base.py:699-702`)。
- `LangchainLLMWrapper`(`RG:llms/base.py:142`)仅在已有 langchain chat model 时才需要;`evaluate()` 会自动包 langchain 对象(`RG:evaluation.py:158-159`)。
- **不传 llm 的后果**:`evaluate()` 默认 `OpenAI()` + `gpt-4o-mini`(`RG:evaluation.py:174-180`)——百炼环境下必挂,必须显式传 llm(或设 `OPENAI_BASE_URL`/`OPENAI_API_KEY` 环境变量)。
- AnswerRelevancy 还需 embeddings:`embedding_factory(provider="openai", model=..., client=..., base_url=...)`(`RG:embeddings/base.py:655-664`)。百炼 embedding 模型名(如 text-embedding-v4)与兼容端点支持情况**待实测**。

**待验证清单**(本次纯研究未调 API):① 百炼 qwen 系对 instructor `Mode.JSON`(response_format)的支持度,不支持则切 `Mode.MD_JSON`;② 百炼 embeddings 兼容接口;③ qwen judge 在 NLI/verdict 任务上的结构化输出稳定性。与 DeepEval 侧「百炼未必返回 logprobs → GEval 回退整数分」(见 [deepeval-geval-explained.md](./deepeval-geval-explained.md))属同类适配问题。

---

## 7. 用理论解释已观察到的实验现象

### 7.1 StepEfficiency 把 emoji/客套话/冗余比较判为 excess(基线 0.75 而非 1.0)

**主因是 rubric 设计,verbosity bias 是放大器。** StepEfficiency 的 judge prompt 本身是对抗性的:

- 「任何 superfluous, speculative, repetitive, or **stylistic** 的步骤都要尽量压低分」(`DE:metrics/step_efficiency/templates/get_execution_efficiency.txt:14-18`);
- 「任何 beautifying 答案的活动(风格编辑、改写) sharply 降分 ≤0.25」(:24-27);
- **0.75 档的定义就是「one small redundant or stylistic step」**(:51-53)——基线 0.75 恰是 judge 忠实执行 rubric 的结果;
- 「拿不准就假设该动作不必要」「adversarial:除非证明最小化,否则给最低分」(:38-40,110)。

理论连接:MT-Bench §3.3 verbosity bias 证明 judge 对**长度/表面形式高度敏感**(91.3% 失败率);此处 rubric 把该敏感性**反向利用**(惩罚修饰),judge 遂把「最终回复的修辞」也当成「执行步骤」来罚——这是 rubric 未区分「trace 动作」与「回复文风」的漏洞,叠加 judge 的表面形式敏感性。G-Eval §4 的自偏好则提醒:qwen judge 评 qwen 生成的客套话,偏差方向未必可预测。
**修法**(按 §1.3/§3 工具箱):rubric 明确「excess 仅指 trace 中的动作,最终回复的礼貌用语/emoji 不计」;用 30–50 条校准集 + few-shot 判例固化边界;必要时交换/复判降方差。

### 7.2 temp 0.7 时同一 case 分数 1.0/0.9/0.6,双 0.0 后可复现

**方差来源分解**:

| 来源 | 机制 | 证据/出处 | 控制手段 |
|---|---|---|---|
| 被测 app 采样方差 | temp 0.7 每次生成不同答案,指标输入本身在变 | MT-Bench §3.3 position bias 实验正是用 temp 0.7 对同一问题生成两个**实质不同**的答案 | app temp → 0 |
| judge 采样方差 | judge 自身解码随机 | G-Eval §3.1 对 GPT-3.5 设 temp 0 「increase determinism」 | judge temp → 0 |
| 打分范式固有波动 | single-answer grading 绝对分天然比 pairwise 不稳 | MT-Bench §3.1 原文 | 概率加权(G-Eval §2,需 logprobs)或 verdict 比例型指标(拆成小判断再代码聚合,降单点波动) |

双 0.0 后消除了前两项采样源,故可复现。注意:verdict 型指标(Faithfulness 等)把语义判断拆成多个小 verdict 再用**代码**算比例,聚合层零方差——这是 QAG 范式相对 GEval 直接打总分的稳定性优势(Ragas 论文 §5 Table 1 的准确率优势同源)。

### 7.3 judge 区分「504 超时被迫重试」(不扣)与「无错误主动重复搜索」(扣)

- **能力面**:judge 能从 trace 里的错误信息做因果归因,这正是 LLM-as-a-judge 的可解释性优势(MT-Bench §3.2:不只给分还给解释)。
- **局限面**:MT-Bench §3.3 证明 judge 推理**脆弱**——连自己会做的数学题都会被上下文带偏;且 StepEfficiency rubric 字面写着「Repetition, re-querying … indicate inefficiency」(模板 :30-32),judge 对 504 网开一面是**超越字面 rubric 的推理**,没有规则保障,换个 case/模型可能翻转。
- **工程化修法**:把「因错误重试不扣分」写进 rubric(显式规则 > 依赖 judge 临场推理);或 reference-guided 思路(MT-Bench §3.4)——给 judge 提供重试策略参考。校准集里应专门放这两类对照案例(§3.2 第 1 步「覆盖已知失败案例」)。

### 7.4 ToolCorrectness 默认是确定性集合对比(不经过 LLM)

源码核实(`DE:metrics/tool_correctness/tool_correctness.py`):

- 默认 `evaluation_params=[]`(:55),`_calculate_non_exact_match_score`(:518-559)按 **name+type 贪心最优匹配**,score = 匹配分之和 / expected 数——纯代码;`should_exact_match` 走全等(:493),`should_consider_ordering` 走加权 LCS(:561)。
- **唯一 LLM 入口**:传了 `available_tools` 才追加 tool_selection judge 分,最终取 **min(确定性分, judge 分)**(:120-132)。不传则全程零 LLM 调用。

**「代码 vs judge」的边界**:结构性事实(调了哪些工具、参数是否相等、集合命中)用代码——零方差、零偏差、零成本;语义质量(该不该调这个工具、答案好不好)才交给 judge。Ragas 的 Non-LLM/ID-based 变体(§5.3)是同一哲学的检索端版本。DeepEval QAG 指标则是**混合边界**:LLM 只做原子 verdict,聚合公式在代码里(`DE:metrics/utils/qag.py:113-138`)——把 judge 的自由度压到最小,正是 §7.2 稳定性与 Ragas 论文 §5 准确率的共同来源。

---

## 8. 参考来源清单

### 论文(一手,arXiv HTML 全文已读)

| 文献 | 版本/出处 | 本文引用节 |
|---|---|---|
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) v4(NeurIPS 2023) | §3.1–3.4,§4.1–4.2,Table 2/3/4/5/13,Appendix D.3/F |
| Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* | [arXiv:2303.16634](https://arxiv.org/abs/2303.16634) v3(EMNLP 2023) | §2,§3.1,§3.4,§4(含三个消融小节) |
| Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation* | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) v1(EACL 2024) | §1,§3,§4,§5 Table 1 |
| Cohen, *A Coefficient of Agreement for Nominal Scales* | Educ. Psychol. Meas. 20(1):37–46, 1960. DOI: 10.1177/001316446002000104 | 原文未直读,动机/公式经 McHugh 2012 核实 |
| Landis & Koch, *The Measurement of Observer Agreement for Categorical Data* | Biometrics 33(1):159–174, 1977. DOI: 10.2307/2529310 | 解读区间经 McHugh 2012 + Langfuse docs 两处转载核实 |
| McHugh, *Interrater reliability: the kappa statistic* | Biochemia Medica 22(3):276–282, 2012. [PMC3900052](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/)(开放获取,全文已读) | κ 公式、区间、局限 |

### 实践指南与官方文档(已抓取核实)

- Hamel Husain, *Creating a LLM-as-a-Judge That Drives Business Results*: <https://hamel.dev/blog/posts/llm-judge/>(critique shadowing、样本量、TPR/TNR、迭代流程;文中引 Shankar et al. *Who Validates the Validators?*,arXiv:2404.12272)
- Langfuse:<https://langfuse.com/guides/llm-as-a-judge-calibration-skill>、<https://langfuse.com/docs/evaluation/scores/score-analytics>、<https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge>;本地 skill `~/.claude/skills/langfuse/references/judge-calibration.md`
- Ragas 官方 docs:<https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/>(变体谱系、ID-based 公式、legacy 弃用时间表)

### Installed 源码(一手,文件:行号见正文)

- DeepEval v4.3.4:`DE:metrics/{faithfulness,answer_relevancy,contextual_precision,contextual_recall}/`、`DE:metrics/utils/qag.py`、`DE:templates/metrics/templates.json`、`DE:metrics/{tool_correctness,step_efficiency}/`
- Ragas 0.4.4:`RG:metrics/collections/{faithfulness,answer_relevancy,context_precision,context_recall}/`、`RG:metrics/_{context_precision,context_recall,string,faithfulness,answer_relevance}.py`、`RG:dataset_schema.py`、`RG:evaluation.py`、`RG:llms/base.py`、`RG:embeddings/base.py`

### 存疑/待验证

1. **百炼适配三项**(§6.3):instructor Mode.JSON 支持度、embeddings 兼容接口、qwen 结构化输出稳定性——未调 API,阶段 2 动手时首验。
2. Landis & Koch 区间为两处权威转载核实,原论文付费墙未直读(区间本身无争议)。
3. MT-Bench self-enhancement bias **论文自认统计上不能定论**,本文按原文限定表述,不作强断言。
4. 两框架分数差异的**定量**验证(同数据双跑)属阶段 2 第 4 步动手内容,本文只给出算法层预测。