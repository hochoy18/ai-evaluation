"""
Agent 轨迹评测最小例(Sync 手动插桩)——学习路径 §4 第 2 步。

与单轮 case 的区别:
- 评估单位是 trace(span 树),不是文本快照;
- Golden 只有题目,答案由 agent 运行时真实产生;
- @observe 生成 span 树,evals_iterator 自动捕获 trace 并跑轨迹指标;
- 这是普通脚本:uv run python 跑,不是 deepeval test run。

运行:
    uv run python src/deepeval_cases/agent_minimal.py
看 span 树(TUI):
    uv run deepeval inspect
"""

import os

from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import OpenAIModel
from deepeval.tracing import observe, update_current_trace

load_dotenv()

# ---------- judge 模型 ----------
judge_model = OpenAIModel(
    model="qwen4-max-preview",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)

# ---------- 被测应用:最小 agent(真实 LLM 调用) ----------
app_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
APP_MODEL = os.getenv("APP_MODEL", "qwen4-max-preview")


@observe(type="llm")  # LLM span:推理层
def call_llm(query: str) -> str:
    response = app_client.chat.completions.create(
        model=APP_MODEL,
        messages=[
            {"role": "system", "content": "你是一个 helpful assistant,回答简洁。"},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


@observe(type="agent")  # agent span:树根,包含上面的 llm span
def my_ai_agent(query: str) -> str:
    answer = call_llm(query)
    # 给树根补上轨迹级测试用例字段(题目 + 最终答案)
    update_current_trace(input=query, output=answer)
    return answer


# ---------- 数据集:只有题目的考卷 ----------
dataset = EvaluationDataset(goldens=[
    Golden(input="为巴黎三日游制定一个简要行程,每天 2-3 个景点"),
    Golden(input="用一句话解释什么是 RAG(检索增强生成)"),
])

# ---------- 跑题循环:Sync 模式,一次一个 ----------
task_completion = TaskCompletionMetric(threshold=0.7, model=judge_model)

for golden in dataset.evals_iterator(
    metrics=[task_completion],
    async_config=AsyncConfig(run_async=False),
):
    rprint(f"[bold cyan]任务:[/bold cyan] {golden.input}")
    my_ai_agent(golden.input)

rprint("[bold green]完成。运行 uv run deepeval inspect 查看 span 树。[/bold green]")
