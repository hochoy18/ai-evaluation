# GEval 详解（deepeval 4.3.4 实测）

针对这段代码：

```python
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=model,
)
```

## 一句话

`GEval` 是 DeepEval 对 **G-Eval 框架**（论文 *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*）的实现：用 **LLM 当裁判**，按你用**自然语言写的任意标准**给输出打分。`correctness_metric` 这里评的是「actual_output 相对 expected_output 是否正确」。

## 逐参数

| 参数 | 本例值 | 作用 |
| --- | --- | --- |
| `name` | `"Correctness"` | 指标标签，用于报告/日志展示，不影响评分逻辑 |
| `criteria` | 自然语言句子 | **核心**：评判标准。裁判 LLM 会先据此自动生成「评估步骤」 |
| `evaluation_params` | `[ACTUAL_OUTPUT, EXPECTED_OUTPUT]` | 告诉裁判**用测试用例的哪些字段**来评。这里把 actual 与 expected 喂给裁判 |
| `threshold` | `0.5` | 及格线。分数 0–1，`score >= threshold` 判通过 |
| `model` | `model`（百炼 OpenAIModel） | 裁判 LLM。不传则用默认（百炼适配见 adapter 文档） |

**隐含默认值**（完整签名里还有）：

- `async_mode=True`：默认异步评测。
- `strict_mode=False`：若设 True，`threshold` 被**强制改为 1**（只有满分才过），分数取整。
- `top_logprobs=20`：概率加权评分时取分数 token 的前 20 个 logprobs（见下）。
- `evaluation_steps=None`：可直接给定评估步骤，跳过「由 criteria 生成步骤」这一步。
- `rubric=None`：可给分档评分细则（每档一个 score_range，0–10 内）。
- `verbose_mode=False`、`flaky=False`。

> 约束：`criteria` 与 `evaluation_steps` 至少给一个；`evaluation_params` 给了就不能是空列表。

## 内部算法（两阶段 + 概率加权）

`assert_test` → `metric.measure(test_case)`（默认走 `a_measure`）：

1. **生成评估步骤**（`_generate_evaluation_steps`）
   若没显式给 `evaluation_steps`，裁判 LLM 根据 `criteria` 自动产出一组有序评估步骤（chain-of-thought）。这是 G-Eval 的「先想清楚怎么评」。

2. **按步骤打分**（`_evaluate`）
   把 `evaluation_params` 指定的字段（actual_output、expected_output）+ 评估步骤 + rubric/ score_range 拼成 prompt，裁判返回 `ReasonScore`（一个整数分 + 理由 reason）。
   - 默认 `score_range = (0, 10)`：裁判打 0–10 的整数分。

3. **概率加权**（`calculate_weighted_summed_score`，G-Eval 的招牌）
   不是直接取裁判吐出的整数，而是取该分数 token 的 `top_logprobs`（默认 20 个候选分数），**按概率加权平均**，得到更细粒度的浮点分。
   - **若无 logprobs，回退为原始整数分**。

4. **归一化到 0–1**
   `score = (g_score - score_range[0]) / (score_range[1] - score_range[0])`，默认即 `g_score / 10`。
   （strict_mode 下直接 `int(g_score)`。）

5. **判通过**：`success = score >= threshold`，并保留 `reason`。

## 关键细节与坑

- **evaluation_params 必须覆盖 criteria 引用的字段**：criteria 说「based on expected output」，就必须含 `EXPECTED_OUTPUT`，否则裁判看不到它。
- **省略 evaluation_params 会改变语义**：源码 `requires_trace = not evaluation_params`。不传 evaluation_params 时，GEval 转为**评 trace（Agent 轨迹）**而非单轮字段——这是它也能做轨迹评测的原因。
- **百炼适配下的评分粒度**：G-Eval 的概率加权依赖裁判返回 `logprobs`。百炼 OpenAI 兼容接口**未必返回 logprobs**，此时回退为原始整数分 → 最终分数粒度是 0, 0.1, …, 1.0（而非连续）。功能正常，只是没那么细。
- **strict_mode**：把及格线顶到 1，适合「必须完全正确才算过」的场景。
- **criteria / evaluation_steps / rubric 三选一或组合**：criteria 最省事（自动生成步骤）；evaluation_steps 更可控；rubric 定义分档标准。

## 运行链路小结

```
assert_test(test_case, [correctness_metric])
  └─ measure → a_measure
       ├─ 由 criteria 生成 evaluation_steps（LLM 调用 1）
       ├─ 按 steps + actual/expected 打分 → ReasonScore（LLM 调用 2）
       ├─ logprobs 概率加权（可用时）→ 归一化 0–1
       └─ score >= 0.5 ? 通过 : 失败（附 reason）
```

每个 GEval 指标至少 2 次裁判 LLM 调用——这也是为什么裁判模型限流时评测会「卡住」。