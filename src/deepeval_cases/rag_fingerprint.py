"""
三模式失败注入指纹实验(学习计划第 3 步,架构 §10.6)。

同一份门禁经正典运行器跑三遍(基线 / --bad-retrieval / --hallucinate),
读每遍的 test_run 快照,聚合四指标均值 + reason 摘录,产出学习基线
指纹对照表。注入模式下门禁失败是预期——评估体系的价值在失败时才显现。

指纹核对用相对模式(暴跌/跌/不动),不断言精确分数;
「暴跌」操作化为 Δ≤−0.2,「不动」操作化为 |Δ|≤0.1(初值,定性判断辅助)。

运行(真实调用 LLM,约 3×9 题×(1 生成+4 裁判)):
    uv run python src/deepeval_cases/rag_fingerprint.py
产出:
    docs/rag-system/rag-fingerprint-medical.md(学习资产,进 git)
"""

import json
import subprocess
from datetime import date
from pathlib import Path

from rich import print as rprint

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_FILE = "src/deepeval_cases/test_rag.py"
SNAPSHOT = REPO_ROOT / ".deepeval" / ".latest_run_full.json"
REPORT = REPO_ROOT / "docs" / "rag-system" / "rag-fingerprint-medical.md"

MODES = [
    ("基线", []),
    ("--bad-retrieval", ["--bad-retrieval"]),
    ("--hallucinate", ["--hallucinate"]),
]
METRICS = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Precision",
    "Contextual Recall",
]
# 相对模式的操作化阈值(初值):暴跌 = 降幅≥0.2;不动 = 变化≤0.1
PLUNGE_DELTA = -0.2
STILL_DELTA = 0.1


def preflight_judge():
    """裁判冒烟:模型不可用(404/403)时大声报错。

    deepeval 会把裁判调用异常静默吞成 score=None(实测),门禁全 FAILED
    却看不到原因——跑三模式前先探一次,失败即终止,不浪费生成端调用。
    """
    from deepeval_cases.test_rag import JUDGE_MODEL, judge

    rprint(f"[bold cyan]裁判冒烟:[/bold cyan] {JUDGE_MODEL} @ token-plan")
    judge.generate("只回复两个字:你好")
    rprint("  裁判可用 ✓")
    return JUDGE_MODEL


def run_mode(flags: list[str]) -> dict:
    """正典运行器跑一遍门禁,返回快照中的门禁用例(注入模式允许门禁失败)。"""
    cmd = ["uv", "run", "deepeval", "test", "run", GATE_FILE, *flags]
    rprint(f"[bold cyan]运行:[/bold cyan] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    # 0=全绿;1=门禁断言失败(注入模式的预期);其余=运行器/收集异常
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"门禁运行器异常(exit {proc.returncode}):\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    cases = [tc for tc in snap["testCases"] if tc["name"].startswith("test_rag_gate")]
    if not cases:
        raise RuntimeError("快照中没有门禁用例——收集可能失败")
    rprint(
        f"  用例 {len(cases)} 条,通过 {snap['testPassed']} / 失败 {snap['testFailed']}"
    )
    return {"cases": cases, "passed": snap["testPassed"], "failed": snap["testFailed"]}


def aggregate(cases: list[dict]) -> dict:
    """每指标:均值 + 最低分用例的 reason 摘录(最低分最具诊断价值)。"""
    stats = {}
    for metric in METRICS:
        entries = [
            (tc["name"], md)
            for tc in cases
            for md in tc["metricsData"]
            if md["name"] == metric and md.get("score") is not None
        ]
        if not entries:
            raise RuntimeError(f"快照缺少指标 {metric} 的分数")
        worst_name, worst_md = min(entries, key=lambda e: e[1]["score"])
        stats[metric] = {
            "mean": sum(md["score"] for _, md in entries) / len(entries),
            "n": len(entries),
            "worst_case": worst_name,
            "worst_score": worst_md["score"],
            "worst_reason": (worst_md.get("reason") or "").strip(),
        }
    return stats


def fmt_delta(value: float, base: float) -> str:
    delta = value - base
    return f"{value:.2f} (Δ{delta:+.2f})"


def build_report(results: dict, judge_model: str) -> str:
    base = results["基线"]["stats"]
    bad = results["--bad-retrieval"]["stats"]
    hal = results["--hallucinate"]["stats"]

    def delta(stats: dict, metric: str) -> float:
        return stats[metric]["mean"] - base[metric]["mean"]

    checks = [
        (
            "bad-retrieval:Contextual Precision 暴跌",
            delta(bad, "Contextual Precision") <= PLUNGE_DELTA,
            f"Δ{delta(bad, 'Contextual Precision'):+.2f}",
        ),
        (
            "bad-retrieval:Contextual Recall 暴跌",
            delta(bad, "Contextual Recall") <= PLUNGE_DELTA,
            f"Δ{delta(bad, 'Contextual Recall'):+.2f}",
        ),
        (
            "hallucinate:Faithfulness 跌",
            delta(hal, "Faithfulness") < 0,
            f"Δ{delta(hal, 'Faithfulness'):+.2f}",
        ),
        (
            "hallucinate:Contextual Precision 不动",
            abs(delta(hal, "Contextual Precision")) <= STILL_DELTA,
            f"Δ{delta(hal, 'Contextual Precision'):+.2f}",
        ),
        (
            "hallucinate:Contextual Recall 不动",
            abs(delta(hal, "Contextual Recall")) <= STILL_DELTA,
            f"Δ{delta(hal, 'Contextual Recall'):+.2f}",
        ),
    ]

    lines = [
        "# 医疗 P0 三模式指纹对照表(学习基线)",
        "",
        f"> 日期:{date.today().isoformat()}。",
        "> **定位:学习资产,不冒充业务基线**——医疗 P0 与语料模式语料不同、题不同,分数不可比(架构 §10.5)。",
        f"> 门面:关键词医疗版(6 篇内嵌源文档,ADR 0001 冻结);裁判:bailian_judge({judge_model}, temp 0.0)。",
        "> 真题集:`data/rag_goldens_medical.json`;门禁集 = single_hop + cross_doc,out_of_domain 剔出四指标。",
        "> 阈值(初值,架构 §10.4):F/AR ≥0.8,CP/CR ≥0.7;基线分布建立后校准。",
        "> 运行器:`deepeval test run`,每模式一遍;注入模式门禁失败是预期。",
        "",
        "## 方法论注记(学习基线)",
        "",
        "- **基线未必全绿**:关键词门面固定返回 top-3,生成端忠实总结全部检索文档,"
        "对窄题干会附加同文档相邻信息 + 医疗免责声明(冻结 prompt「不确诊」驱动),"
        "AnswerRelevancy 视之为偏题系统性扣分(实测 g002/g003/g005/g007 AR 0.5–0.75)。"
        "这是门面特征,非门禁代码缺陷;「全绿」验收按学习基线定位重新对齐(issue #2)。",
        "- **temp 0 非确定性**:qwen4-max-preview(生成)/qwen3.7-plus(裁判)为推理/preview 模型,"
        "temp 0 不保证逐字确定,答案在聚焦/冗长间波动 → 门禁跨运行偶发翻转。"
        "指纹核对用相对模式(暴跌/跌/不动)而非精确分数,正是为此鲁棒。",
        "- **超时修复**:deepeval 默认 180s/task,慢端点下 Faithfulness 超时被静默取消(score=None);"
        "设 `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600` 消除该噪声。",
        "- **g006 题干修订**:原窄题干 AR=0.5,加「根据挂号指南…只列情况即可」显式收敛 → AR=1.0,"
        "检索与 expected_output 不变。",
        "",
        "## 分数对照表(门禁集均值)",
        "",
        "| 模式 | Faithfulness | Answer Relevancy | Contextual Precision | Contextual Recall | 门禁 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, _ in MODES:
        stats = results[mode]["stats"]
        cells = []
        for metric in METRICS:
            mean = stats[metric]["mean"]
            cells.append(fmt_delta(mean, base[metric]["mean"]) if mode != "基线" else f"{mean:.2f}")
        gate = f"{results[mode]['passed']} 过 / {results[mode]['failed']} 败"
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {gate} |")

    lines += [
        "",
        "## 指纹核对(架构 §10.6,相对模式)",
        "",
        "预期:bad-retrieval → CP/CR 暴跌、F 可能依然高(忠实≠正确,病灶在检索端);"
        "hallucinate → F 跌、CP/CR 不动(检索没病,病在生成端)。",
        "",
    ]
    for label, ok, evidence in checks:
        lines.append(f"- {'✓' if ok else '✗'} {label}({evidence})")
    bad_f = bad["Faithfulness"]["mean"]
    lines.append(
        f"- bad-retrieval 下 Faithfulness = {bad_f:.2f}"
        "(「可能依然高」不作断言:照错误文档答或尽责拒答,两种形态病灶都在检索端)"
    )

    lines += ["", "## reason 摘录(每模式×指标取最低分用例)"]
    for mode, _ in MODES:
        lines += ["", f"### {mode}"]
        stats = results[mode]["stats"]
        for metric in METRICS:
            s = stats[metric]
            reason = s["worst_reason"]
            if len(reason) > 300:
                reason = reason[:300] + "…"
            case = s["worst_case"].split("[")[-1].rstrip("]")
            lines.append(f"- **{metric}**({case},{s['worst_score']:.2f}):{reason}")

    lines += [
        "",
        "## 逐题分数(附录)",
        "",
        "| 真题 | 模式 | F | AR | CP | CR |",
        "|---|---|---|---|---|---|",
    ]
    for mode, _ in MODES:
        for tc in results[mode]["cases"]:
            case = tc["name"].split("[")[-1].rstrip("]")
            scores = {md["name"]: md.get("score") for md in tc["metricsData"]}
            cells = [
                f"{scores[m]:.2f}" if scores.get(m) is not None else "—" for m in METRICS
            ]
            lines.append(f"| {case} | {mode} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "uv run python src/deepeval_cases/rag_fingerprint.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    judge_model = preflight_judge()
    results = {}
    for mode, flags in MODES:
        run = run_mode(flags)
        results[mode] = {**run, "stats": aggregate(run["cases"])}
    report = build_report(results, judge_model)
    REPORT.write_text(report, encoding="utf-8")
    rprint(f"\n[bold green]指纹对照表已产出:[/bold green] {REPORT}")
    short = {"Faithfulness": "F", "Answer Relevancy": "AR", "Contextual Precision": "CP", "Contextual Recall": "CR"}
    for mode, _ in MODES:
        means = "  ".join(
            f"{short[m]}={results[mode]['stats'][m]['mean']:.2f}" for m in METRICS
        )
        rprint(f"  {mode}: {means}")


if __name__ == "__main__":
    main()