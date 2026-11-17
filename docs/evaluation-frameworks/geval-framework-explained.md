# G-Eval 框架讲解

> 论文：*G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*（Liu et al., 2023，微软，EMNLP 2023，arXiv:2303.16634）
> DeepEval 的 `GEval` 指标即此框架的工程实现（见 [geval-explained](./deepeval-geval-explained.md)）。

## 1. 要解决什么问题

NLG（自然语言生成）评测长期两难：

- **传统自动指标**（BLEU、ROUGE、METEOR 等 n-gram 重叠，BERTScore、MoverScore 等语义相似度）与**人类判断相关性低**——它们衡量「字面/语义相似」，而非「质量好坏」。
- **人工评测**准但**贵、慢、不可扩展**。

G-Eval 的思路：用**强 LLM（GPT-4）当裁判**，配合一套提示与评分方法，让自动评测**逼近人类判断**。

## 2. 三大组件

### ① Prompt：任务定义 + 评估标准

给裁判 LLM 一段提示，包含：

- 任务介绍（如「这是一个摘要任务」）。
- 评估维度与标准。论文以摘要为例定义了四个维度（各 1–5 分）：
  - **Coherence（连贯性）**：所有句子的整体质量与逻辑流。
  - **Consistency（一致性）**：与源文的事实对齐，有无 hallucination。
  - **Fluency（流畅度）**：单句的语法与可读性。
  - **Relevance（相关性）**：是否选取了源文的重要内容。

> 对应 DeepEval：这就是你写的 `criteria`（自然语言标准）。

### ② Chain-of-Thought：自动生成评估步骤

不直接让 LLM 打分，而是先让它**根据标准生成一组详细的评估步骤**（CoT），再按步骤评。

- 好处：把「怎么评」显式化、结构化，减少随意性，提升与人类的一致性。

> 对应 DeepEval：`_generate_evaluation_steps`——由 `criteria` 自动产出 `evaluation_steps`（第 1 次 LLM 调用）。你也可以直接传 `evaluation_steps` 跳过这步。

### ③ 评分函数：填表范式 + 概率加权（核心创新）

**填表范式（form-filling）**：把评测设计成「让 LLM 在一个评分表单里填一个分数」，而非自由生成文本。这样输出被约束成一个**分数 token**，便于取它的概率。

**概率加权评分**：G-Eval 的招牌。不直接取解码出的那个整数分，而是利用该分数 token 的**概率分布**，对候选分数做加权平均：

```
score = Σ_i  p(s_i) · s_i
```

- `s_i`：候选分数（如 1,2,3,4,5）。
- `p(s_i)`：裁判模型给分数 token `s_i` 的（归一化）概率。

**为什么有效**：

- **细粒度**：把离散整数分变成连续分（例如模型在 3 和 4 之间犹豫，加权后得 3.4），更贴近人类的细腻判断。
- **降方差/更稳**：利用概率信息，比单次解码更稳定，显著提升与人类评分的相关性——对较弱模型尤其明显。

> 对应 DeepEval：`calculate_weighted_summed_score` + `top_logprobs=20`，默认分范围 0–10，最后归一化到 0–1。

## 3. 论文主要结论

- 在摘要（**SummEval**）与对话生成（**Topical-Chat**、**PERSONA-CHAT**）基准上，**G-Eval + GPT-4 与人类判断的相关性高于**此前各类自动指标（n-gram、嵌入相似度等）。
- **概率加权**相比「直接取整数分」明显提升相关性；**CoT** 也有帮助。
- 强裁判模型（GPT-4）是关键——裁判越弱，效果越打折。

## 4. 局限与偏差（务必知道）

- **偏好 LLM 生成文本**：论文指出 G-Eval（GPT-4 裁判）倾向给 **LLM 生成的文本打更高分**，存在「自偏好 / 冗长偏差」。评 LLM 输出时要警惕这种系统性偏高。
- **依赖 logprobs**：概率加权需要 API 返回 token 概率。**拿不到 logprobs 就退化为整数分**，失去细粒度优势。
- **成本与延迟**：每条评测要多次 LLM 调用（生成步骤 + 打分），批量跑要考虑裁判模型的配额/限流。
- **裁判依赖**：结论绑定裁判模型能力；换弱裁判，相关性下降。

## 5. DeepEval 落地对照

| G-Eval 概念 | DeepEval `GEval` |
| --- | --- |
| 任务 + 标准（Prompt） | `criteria`（或 `evaluation_steps` / `rubric`） |
| CoT 评估步骤 | `_generate_evaluation_steps`（自动生成） |
| 填表打分 | `_evaluate` → `ReasonScore`（默认 0–10 整数 + reason） |
| 概率加权 | `calculate_weighted_summed_score`，`top_logprobs=20` |
| 分数 | 归一化 `(g_score - lo)/(hi - lo)` → 0–1，`threshold` 判通过 |

## 6. 对本项目（百炼裁判）的实践启示

- **百炼兼容接口未必返回 logprobs** → G-Eval 的概率加权可能不生效，回退整数分，分数粒度变成 0, 0.1, …, 1.0。功能正常，只是少了「细粒度」这一核心优势。
- **criteria 要写清楚、可操作**：它直接决定自动生成的评估步骤质量。
- **裁判模型选强一点的**（qwen-max 比 qwen-plus 更稳），并温度设 0 降低随机性。
- **警惕自偏好偏差**：用 LLM 评 LLM 输出，分数可能系统性偏高；必要时用人工抽检校准，或引入 `expected_output` 作锚点（本例 correctness 正是如此）。
- **批量跑注意配额**：每个 GEval ≥ 2 次裁判调用，数据集大时易触发百炼限流（评测「卡住」的常见原因）。