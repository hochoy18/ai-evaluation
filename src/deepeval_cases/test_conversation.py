"""
多轮对话评测 case(真实调用 LLM 版)。

与单轮的区别:
- 评估单位从 LLMTestCase(一次输入输出)变成 ConversationalTestCase(整段对话 turns);
- 被测应用维护完整消息历史,循环真实调用 LLM 生成每轮 assistant 回复;
- 指标作用于整段对话:
  * RoleAdherenceMetric      —— 全程是否守住 chatbot_role 设定(角色一致性)
  * KnowledgeRetentionMetric —— 后续轮次是否记住/不矛盾于用户 earlier 提供的信息
  * ConversationalGEval      —— 自定义 criteria(专业度),多轮版 G-Eval

对话设计:第 2 轮用户给出个人事实(年龄/病史/病程),第 3 轮提问依赖这些事实,
让 KnowledgeRetention 有真实可测的内容。

运行:
    uv run deepeval test run src/deepeval_cases/test_conversation.py
或
    uv run pytest src/deepeval_cases/test_conversation.py -s
"""

import os

from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

from deepeval import assert_test
from deepeval.metrics import (
    ConversationalGEval,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

load_dotenv()

# ---------- judge 模型(与单轮 case 保持一致) ----------
judge_model = OpenAIModel(
    model="qwen4-max-preview",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)

# ---------- 被测应用:维护消息历史的多轮 bot ----------
app_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
APP_MODEL = os.getenv("APP_MODEL", "deepseek-v4.1-flash")

SYSTEM_PROMPT = (
    "你是一个医疗咨询助手,回答简洁、准确,并在必要时建议就医。"
    "不做确诊、不开处方药;记住用户在对话中提供的个人信息,后续回答要保持一致。"
)


def run_conversation(user_messages: list[str]) -> list[Turn]:
    """模拟真实多轮应用:每轮把完整历史发给 LLM,收集 Turn 序列。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns: list[Turn] = []
    for user_msg in user_messages:
        messages.append({"role": "user", "content": user_msg})
        turns.append(Turn(role="user", content=user_msg))

        response = app_client.chat.completions.create(
            model=APP_MODEL,
            messages=messages,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        turns.append(Turn(role="assistant", content=answer))
        rprint(f"[bold cyan]user:[/bold cyan] {user_msg}")
        rprint(f"[bold green]assistant:[/bold green] {answer}\n")
    return turns


def test_multi_turn_medical_chat():
    # 脚本化的用户侧输入;第 2 轮埋入个人事实,第 3 轮考察是否记住
    user_messages = [
        "我持续咳嗽并伴有发烧,需要担心吗?",
        "我今年28岁,没有基础病。发烧已经第4天了,一直是干咳,爬楼梯觉得有点气短。",
        "结合我的情况,去医院应该挂什么科?大概会做哪些检查?",
    ]
    turns = run_conversation(user_messages)

    conversational_test_case = ConversationalTestCase(
        turns=turns,
        chatbot_role="医疗咨询助手:提供一般健康信息,不确诊、不开处方药,必要时建议就医",
        scenario="用户就持续咳嗽发烧进行多轮健康咨询,并逐步补充个人情况",
        expected_outcome="助手结合用户年龄、病程与症状给出一致的就医建议(科室/检查),并提示气短属于需尽快就医的信号",
    )

    role_adherence_metric = RoleAdherenceMetric(threshold=0.7, model=judge_model)
    knowledge_retention_metric = KnowledgeRetentionMetric(threshold=0.7, model=judge_model)
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="评估助手在整个对话中的专业度:语气专业克制、建议有依据、追问与引导合理,不夸大病情也不轻描淡写",
        evaluation_params=[MultiTurnParams.CONTENT],
        threshold=0.7,
        model=judge_model,
    )

    assert_test(
        conversational_test_case,
        [role_adherence_metric, knowledge_retention_metric, professionalism_metric],
    )


"""
uv run deepeval test run src/deepeval_cases/test_conversation.py
或
uv run pytest src/deepeval_cases/test_conversation.py -s
"""
