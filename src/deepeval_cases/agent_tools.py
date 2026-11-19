"""
Agent 工具调用评测 + 失败注入实验 —— 学习路径 §4 第 3 步。

树形态:agent → llm(决策) → tool(执行) → llm → tool → llm(最终回复)
两层指标:
  - 轨迹级:TaskCompletion + StepEfficiency,挂 evals_iterator,读整棵树;
  - 组件级:ToolCorrectness + ArgumentCorrectness,挂做出工具决策的 llm span。

三种实验模式:
  基线          正常 search → book;
  --fail-search 首次搜索抛 504,agent 被迫重试(错误恢复);
  --wasteful    注入坏策略:无错误也重复搜索「核对」(主动冗余)。

实测结论:
  - 被迫重试:judge 认为合理,StepEfficiency 不降(0.75→0.75);
  - 主动冗余:StepEfficiency 下降 + 冗余决策 span 的 ToolCorrectness 下降
    ——「答案对,过程烂」,轨迹分说病了,组件分指出病灶。

运行:
    uv run python src/deepeval_cases/agent_tools.py               # 基线
    uv run python src/deepeval_cases/agent_tools.py --fail-search # 失败注入
    uv run python src/deepeval_cases/agent_tools.py --wasteful    # 主动冗余
    uv run deepeval inspect                                       # 看 span 树
"""

import json
import os
import sys
from datetime import date, timedelta

from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_span, update_current_trace

load_dotenv()

# 被 test_agent.py import 时 sys.argv 是 pytest 的参数,两个开关都为 False → 基线模式,
# 这正是 CI 门禁想要的:只测正常路径。失败注入只在脚本形态手动实验。
FAIL_FIRST_SEARCH = "--fail-search" in sys.argv
WASTEFUL = "--wasteful" in sys.argv  # 主动冗余:无错误也重复搜索,制造「答案对,过程烂」
search_call_count = 0

judge_model = OpenAIModel(
    model="qwen4-max-preview",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)
app_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
APP_MODEL = os.getenv("APP_MODEL", "qwen4-max-preview")

# ---------- 工具(action 层) ----------

@observe(type="tool")
def search_flights(origin: str, destination: str, date: str):
    """搜索航班。--fail-search 模式下第一次调用故意抛错。"""
    global search_call_count
    search_call_count += 1
    if FAIL_FIRST_SEARCH and search_call_count == 1:
        raise RuntimeError("航班搜索 API 超时 (504)")
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]


@observe(type="tool")
def book_flight(flight_id: str):
    """按航班 ID 订票。"""
    return {"confirmation": "CONF-789", "flight_id": flight_id}


TOOL_IMPLS = {"search_flights": search_flights, "book_flight": book_flight}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索两个城市之间某天的可用航班,返回航班 ID 与价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "按航班 ID 预订机票,返回确认号",
            "parameters": {
                "type": "object",
                "properties": {"flight_id": {"type": "string"}},
                "required": ["flight_id"],
            },
        },
    },
]

# ---------- LLM 决策层(组件指标挂这里) ----------

@observe(type="llm")
def call_llm(messages: list):
    response = app_client.chat.completions.create(
        model=APP_MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        temperature=0.0,
    )
    msg = response.choices[0].message

    # 只在「做出工具调用决策」的 span 上挂组件指标并提供数据
    if msg.tool_calls:
        called = [
            ToolCall(
                name=tc.function.name,
                input_parameters=json.loads(tc.function.arguments),
            )
            for tc in msg.tool_calls
        ]
        # 期望工具按阶段推断:还没搜过 → 应搜索;搜过了 → 应订票
        searched = any(
            m.get("role") == "tool" and "price" in str(m.get("content", ""))
            for m in messages
        )
        expected = [ToolCall(name="book_flight" if searched else "search_flights")]
        # update_current_span(
        #     tools_called=called,
        #     expected_tools=expected,
        #     metrics=[
        #         ToolCorrectnessMetric(threshold=0.7, model=judge_model),
        #         ArgumentCorrectnessMetric(threshold=0.7, model=judge_model),
        #     ],
        # )
    return msg


# ---------- Agent 循环 ----------

@observe(type="agent")
def travel_agent(user_input: str) -> str:
    # 真实 agent 都会在 system prompt 注入当前日期,否则「明天」无法解析,
    # LLM 会反问用户而不调工具(基线第一次跑就栽在这:TaskCompletion 0.05)。
    today = date.today()
    tomorrow = today + timedelta(days=1)
    system = f"你是机票预订助手。今天是 {today.isoformat()}。用户说「明天」即 {tomorrow.isoformat()}。用工具搜索航班并预订最便宜的一班,最后告知用户航班号、价格与确认号。工具报错时可重试。"
    if WASTEFUL:
        # 故意注入坏策略:无错误也重复搜索「核对」,制造非被迫的冗余步骤
        system += "为了保险,每次搜索后必须用完全相同的参数再搜索一次核对结果,两次一致才预订。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    tools_called = []
    final = ""
    for _ in range(6):  # 最多 6 步,防死循环
        msg = call_llm(messages)
        if not msg.tool_calls:
            final = msg.content or ""
            break
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = TOOL_IMPLS[tc.function.name](**args)
                out = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                out = f"工具调用失败: {e}"  # 错误回喂给 LLM,让它自行恢复
            tools_called.append(ToolCall(name=tc.function.name, input_parameters=args, output=out))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    #
    # update_current_trace(
    #     input=user_input,
    #     output=final,
    #     tools_called=tools_called,
    #     expected_tools=[ToolCall(name="search_flights"), ToolCall(name="book_flight")],
    # )
    return final


# ---------- 数据集 + 跑题循环(仅脚本形态;import 时不执行) ----------

if __name__ == "__main__":
    dataset = EvaluationDataset(goldens=[
        Golden(input="订明天从纽约到洛杉矶最便宜的航班"),
    ])

    mode = "失败注入(首次搜索抛错)" if FAIL_FIRST_SEARCH else ("主动冗余(重复搜索核对)" if WASTEFUL else "基线")
    rprint(f"[bold cyan]实验模式:[/bold cyan] {mode}")

    for golden in dataset.evals_iterator(
        metrics=[
            TaskCompletionMetric(threshold=0.7, model=judge_model),
            StepEfficiencyMetric(threshold=0.7, model=judge_model),
        ],
        async_config=AsyncConfig(run_async=False),
    ):
        rprint(f"[bold cyan]任务:[/bold cyan] {golden.input}")
        answer = travel_agent(golden.input)
        rprint(f"[bold green]最终回复:[/bold green] {answer}")

    rprint("[bold]运行 uv run deepeval inspect 查看 span 树与逐 span 分数。[/bold]")
