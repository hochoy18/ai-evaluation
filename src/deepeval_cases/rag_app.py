"""
最小真实 RAG app —— RAG 线第 1 步(检索 + 生成,带插桩)。

与 agent 线的关键区别:RAG 四指标是单轮指标,读 LLMTestCase 的字段
(retrieval_context 等),不依赖 trace。这里的 @observe 是为了可观测性
(inspect 看 retriever span)与以后在线评估铺路,不是指标的前提。

结构:
    rag_app (agent span,树根)
    ├── retrieve (retriever span)  关键词命中打分,top-k 有序 list[str]
    └── generate (llm span)        只依据参考资料回答,temp=0.0

失败钩子(第 3 步用,被 pytest import 时自动为 False → 基线):
    --bad-retrieval  检索器强制返回无关文档(污染检索端)
    --hallucinate    生成 prompt 翻转为自由发挥(污染生成端)

运行:
    uv run python src/deepeval_cases/rag_app.py
    uv run deepeval inspect
"""

import os
import sys

from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

from deepeval.tracing import observe, update_current_span, update_current_trace

load_dotenv()

BAD_RETRIEVAL = "--bad-retrieval" in sys.argv
HALLUCINATE = "--hallucinate" in sys.argv

app_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
APP_MODEL = os.getenv("APP_MODEL", "qwen4-max-preview")

# ---------- 知识库:6 篇短文档,主题边界清晰,事实具体(供 claims 验证) ----------

DOCS = [
    {
        "id": "fever",
        "keywords": ["发烧", "发热", "体温", "退烧", "高热", "低烧"],
        "text": "发烧通常指腋下体温≥37.3℃,38.5℃及以上属中高热。成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医。低烧可先物理降温、多饮水;退烧药需在医生指导下使用,两次用药间隔至少4-6小时。",
    },
    {
        "id": "cough",
        "keywords": ["咳嗽", "干咳", "咳痰", "痰", "气短", "呼吸"],
        "text": "咳嗽分干咳(无痰)与湿咳(有痰)。干咳持续超过2周,或伴随气短、咯血、胸痛,应尽快到呼吸内科检查。爬楼气短是下呼吸道或心肺受累的警示信号,不可忽视。湿咳需观察痰色与痰量,黄脓痰可能提示细菌感染。",
    },
    {
        "id": "hypertension",
        "keywords": ["高血压", "血压", "降压", "头晕"],
        "text": "高血压诊断标准:未服降压药情况下,非同日三次测量收缩压≥140mmHg和/或舒张压≥90mmHg。生活方式干预包括限盐(每日<5克)、规律运动、控制体重、戒烟限酒。确诊后需长期监测并遵医嘱规范治疗,不可自行停药。",
    },
    {
        "id": "diabetes",
        "keywords": ["糖尿病", "血糖", "多饮", "多尿", "体重下降"],
        "text": "糖尿病典型症状为「三多一少」:多饮、多尿、多食、体重下降。诊断标准:空腹血糖≥7.0mmol/L,或餐后2小时血糖≥11.1mmol/L,或糖化血红蛋白≥6.5%。高危人群应定期筛查血糖,管理包括饮食控制、运动与遵医嘱用药。",
    },
    {
        "id": "registration",
        "keywords": ["挂", "科", "挂号", "就诊", "急诊", "呼吸内科", "心内科", "内分泌"],
        "text": "挂号指南:发烧、咳嗽等呼吸道症状挂呼吸内科;高血压复诊挂心内科或普通内科;糖尿病挂内分泌科。症状不明确可先咨询导诊台。突发高热伴意识改变、呼吸困难、剧烈胸痛等急重症直接去急诊。就诊请携带既往检查报告与病历。",
    },
    {
        "id": "medication",
        "keywords": ["药", "用药", "处方", "抗生素", "退烧药", "布洛芬"],
        "text": "用药安全:处方药须凭医生处方使用,抗生素不可自行服用。退烧药(如布洛芬、对乙酰氨基酚)两次间隔至少4-6小时,24小时不超过4次;不要叠加多种复方感冒药,避免相同成分过量。自行用药3天症状不缓解应就医。孕妇、儿童、肝肾功能不全者用药须遵医嘱。",
    },
]

# ---------- 检索端 ----------

@observe(type="retriever")
def retrieve(question: str, k: int = 3) -> list[str]:
    """关键词命中打分,返回 top-k 有序 list[str](排名即数据,ContextualPrecision 按序加权)。"""
    if BAD_RETRIEVAL:
        # 失败注入:问什么都返回无关主题(高血压/糖尿病/用药),污染检索端
        bad = [d["text"] for d in DOCS if d["id"] in ("hypertension", "diabetes", "medication")]
        update_current_span(retrieval_context=bad)
        return bad
    scored = [(sum(kw in question for kw in d["keywords"]), d) for d in DOCS]
    scored.sort(key=lambda x: -x[0])  # 稳定排序:同分按 DOCS 原顺序
    top = [d["text"] for _, d in scored[:k]]
    update_current_span(retrieval_context=top)
    return top


# ---------- 生成端 ----------

SYSTEM_BASE = (
    "你是医疗咨询助手。只依据下方参考资料回答:资料不足时明确说明「资料中没有相关信息」,不要编造。"
    "不确诊、不开处方药。回答简洁,可引用资料编号(如[资料1])。"
)
SYSTEM_HALLUCINATE = (
    "你是医疗咨询助手。自由发挥,不必拘泥于参考资料,可以补充你自己的医学知识和建议。"
    "回答尽量详尽丰富。"
)


@observe(type="llm")
def generate(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    response = app_client.chat.completions.create(
        model=APP_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_HALLUCINATE if HALLUCINATE else SYSTEM_BASE},
            {"role": "user", "content": f"参考资料:\n{context_block}\n\n用户问题:{question}"},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ---------- 顶层流水线 ----------

@observe(type="agent")
def rag_app(question: str) -> tuple[str, list[str]]:
    """返回 (答案, 检索上下文)——测试要同时拿两者组装 LLMTestCase。"""
    contexts = retrieve(question)
    answer = generate(question, contexts)
    update_current_trace(input=question, output=answer, retrieval_context=contexts)
    return answer, contexts


# ---------- demo(仅脚本形态;import 时不执行) ----------

if __name__ == "__main__":
    mode = "检索污染" if BAD_RETRIEVAL else ("生成污染" if HALLUCINATE else "基线")
    rprint(f"[bold cyan]实验模式:[/bold cyan] {mode}")

    for q in [
        "发烧四天干咳挂什么科",
        "血压150/95算高血压吗,平时要注意什么",
    ]:
        answer, contexts = rag_app(q)
        rprint(f"\n[bold cyan]问题:[/bold cyan] {q}")
        rprint(f"[bold]检索命中({len(contexts)} 篇,按排名):[/bold]")
        for i, c in enumerate(contexts):
            rprint(f"  {i+1}. {c[:50]}…")
        rprint(f"[bold green]答案:[/bold green] {answer}")
