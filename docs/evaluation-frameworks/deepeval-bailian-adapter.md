# DeepEval 适配百炼（Bailian / OpenAI 兼容接口）

> 适用版本：deepeval **4.3.4**（本项目实测）。endpoint 示例：
> `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`

## 核心：改三样东西

DeepEval 的 LLM-as-a-judge 指标（GEval、TaskCompletionMetric 等）默认调用 OpenAI 的 gpt 模型。换成百炼只需：

1. **base_url** → 百炼兼容 endpoint
2. **api_key** → 百炼 API Key
3. **模型名** → 百炼模型（`qwen-plus` / `qwen-max` / `qwen-turbo` 等，须为你的套餐支持）

> ⚠️ 只改 base_url 不改模型名会失败：默认裁判模型是 gpt 系列（4.3.4 为 `gpt-5.4`），百炼没有这个模型。

## 方式一：显式传 model（推荐，清晰可控）

用 `deepeval.models.OpenAIModel`（`GPTModel` 是其弃用别名），它原生支持 `base_url`：

```python
from deepeval.models import OpenAIModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

judge = OpenAIModel(
    model="qwen-plus",
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key="sk-你的百炼key",
    temperature=0.0,
)

metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge,          # 关键：把裁判模型传进去
    threshold=0.5,
)
```

本项目已封装好工厂函数 `ai_evaluation.models.bailian_judge`：

```python
from ai_evaluation.models import bailian_judge
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

metric = GEval(
    name="Correctness",
    criteria="...",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=bailian_judge("qwen-plus"),
    threshold=0.5,
)
```

`bailian_judge` 从环境变量读 base_url（`OPENAI_BASE_URL`）与 key（`BAILIAN_API_KEY` 优先，其次 `OPENAI_API_KEY`）。

## 方式二：全局环境变量（设一次，所有指标生效）

DeepEval 的 settings（pydantic BaseSettings）按字段名读环境变量。设：

```bash
OPENAI_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-你的百炼key
OPENAI_MODEL_NAME=qwen-plus      # 默认裁判模型名（关键）
USE_OPENAI_MODEL=true            # 显式选 OpenAI 供应商（可不设，默认就回退到它）
```

原理（4.3.4 源码核实）：

- 指标 `model=None` → `initialize_model` → `OpenAIModel(model=None)`。
- `OpenAIModel` 里 `model = model or settings.OPENAI_MODEL_NAME`，再回退默认 gpt。所以 `OPENAI_MODEL_NAME=qwen-plus` 生效。
- `OpenAIModel.base_url` 未显式传时为 `None`，传给 OpenAI SDK 后由 **SDK 自己读环境变量 `OPENAI_BASE_URL`**（DeepEval settings 里**没有** OPENAI_BASE_URL 字段，靠 SDK 回退）。
- api_key 读 `settings.OPENAI_API_KEY`（即 env `OPENAI_API_KEY`）。

本项目 `.env.example` 已含上述变量；`cp .env.example .env.local` 后填 key 即可（DeepEval import 时自动加载 .env.local/.env）。

> CLI 备注：`deepeval set-openai --model qwen-plus` 也能设默认模型名，但它**没有 --base-url 选项**，且对自定义模型要求填 `--cost-per-input-token/--cost-per-output-token`。所以百炼适配更推荐上面两种方式（base_url 走 env 或显式 OpenAIModel）。

## 关于成本（cost）

- 未知模型（qwen-*）会回退到默认 `DeepEvalModelData`，**Python 侧不强制**填成本——不填只是不做成本统计，不影响评测。
- 想统计成本就传 `cost_per_input_token` / `cost_per_output_token`（或对应 env）。

## 裁判模型选择建议

- 评测裁判需要较好的指令遵循与 JSON/结构化输出能力：`qwen-plus` 性价比高，`qwen-max` 更稳。
- 温度建议 0（`bailian_judge` 默认 0），让打分更确定。
- 若指标卡住，多半是裁判模型限流/超额/网络（见 getting-started FAQ），先查百炼配额与 key。

## 验证（无需真实 key 即可检查构造）

```bash
uv run python -c "from ai_evaluation.models import bailian_judge; m=bailian_judge('qwen-plus'); print(type(m).__name__, m.model, m.base_url)"
```

真实跑通需有效百炼 key：设好 env 后 `uv run deepeval test run tests/test_example.py`。