"""
Ingestion CLI 缝 —— hermetic 切片(架构 §4–§6,issue #3 验收 7)。

只经两个公共缝观察外部产物,内部组件无直接单测:
  1. ingest CLI 缝:``rag_core.ingest.main(argv, embedder=...)``
     (Embedder 协议替换位注入确定性 fake,零网络零计费);
     外部产物 = Chroma collection 内容 / manifest JSON / stdout 报告。
  2. 存储层查询门禁缝:``rag_core.store.open_query_collection(config)``
     (vector 后端票才接门面;此处验证存储层拒查本身)。

运行(裸 pytest 即可,本文件不碰网络):
    uv run pytest src/rag_core/test_ingest_seam.py -q
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from rag_core.embeddings import FakeEmbedder
from rag_core.ingest import main

# 期望值独立来源:架构 §4.4(doc_id = sha1(相对路径)[:12])、§6.2(collection 名)
COLLECTION_NAME = "rag_corpus"


def doc_id_of(relpath: str) -> str:
    return hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12]


BASIC_MD = """# 就诊指南

## 发烧处理

### 体温判定

腋下体温≥37.3℃即属发烧,38.5℃及以上属中高热。低烧可先物理降温、多饮水,并持续观察体温变化与伴随症状。成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医,不要自行加量用药。退烧药需在医生指导下使用,两次用药间隔至少4-6小时,24小时不超过4次,儿童与孕妇用药须遵医嘱。
"""


@dataclass
class IngestRun:
    exit_code: int
    stdout: str
    corpus_dir: Path
    index_dir: Path
    manifest_file: Path
    embedder: FakeEmbedder

    def manifest(self) -> dict | None:
        if not self.manifest_file.exists():
            return None
        return json.loads(self.manifest_file.read_text(encoding="utf-8"))

    def collection(self):
        # 与 CLI 侧同一 settings(遥测关闭),否则 Chroma 缓存按 settings 区分报冲突
        client = chromadb.PersistentClient(
            path=str(self.index_dir), settings=Settings(anonymized_telemetry=False)
        )
        return client.get_collection(COLLECTION_NAME)


def run_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    files: dict[str, str],
    argv: tuple[str, ...] = (),
    goldens: list[dict] | None = None,
    env_extra: dict[str, str] | None = None,
) -> IngestRun:
    """布置 tmp 语料/索引/manifest/真题,经 env 注入配置,跑 CLI 缝。"""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    for rel, text in files.items():
        p = corpus_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    index_dir = tmp_path / "index"
    manifest_file = tmp_path / "rag_manifest.json"
    goldens_file = tmp_path / "goldens.json"
    if goldens is not None:
        goldens_file.write_text(json.dumps(goldens, ensure_ascii=False), encoding="utf-8")

    env = {
        "RAG_CORPUS_DIR": str(corpus_dir),
        "RAG_INDEX_DIR": str(index_dir),
        "RAG_MANIFEST_FILE": str(manifest_file),
        "RAG_GOLDENS_FILE": str(goldens_file),
        "EMBEDDING_BACKEND": "fake",
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIM": "1024",
    }
    env.update(env_extra or {})
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    embedder = FakeEmbedder(dim=int(env["EMBEDDING_DIM"]))
    exit_code = main(list(argv), embedder=embedder)
    stdout = capsys.readouterr().out
    return IngestRun(exit_code, stdout, corpus_dir, index_dir, manifest_file, embedder)


def test_batch_cap_and_usage_cost_log(tmp_path, monkeypatch, capsys):
    """batch >10 永不发出:25 chunks → 3 次请求;token 用量与成本估算入日志。"""
    sections = "\n\n".join(
        f"### 小节{i:02d}\n\n第{i:02d}节独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化。成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医。退烧药需在医生指导下使用,两次用药间隔至少4-6小时,24小时不超过4次。儿童与孕妇用药须遵医嘱。"
        for i in range(25)
    )
    md = "# 大文档\n\n## 章节\n\n" + sections + "\n"
    run = run_ingest(tmp_path, monkeypatch, capsys, {"big.md": md})
    assert run.exit_code == 0, run.stdout
    assert run.collection().count() == 25
    assert run.embedder.usage.requests == 3  # ceil(25/10),batch ≤10 强制
    assert "[embedding] 25 chunks, 3 次请求(batch≤10)" in run.stdout
    assert "tokens=" in run.stdout and "成本≈¥" in run.stdout
    assert run.manifest()["embedding_dim"] == 1024


def _golden(gid: str, doc_ids: list[str], difficulty: str = "single_hop") -> dict:
    return {
        "id": gid,
        "question": f"{gid} 的问题文本",
        "expected_output": "期望答案",
        "expected_doc_ids": doc_ids,
        "difficulty": difficulty,
        "source": "synthetic",
        "notes": "保鲜测试 fixture",
    }


def test_freshness_lists_affected_goldens_without_mutating_them(tmp_path, monkeypatch, capsys):
    """真题保鲜:变更/删除源文档 ∩ expected_doc_ids → 清单供人工 review;真题文件永不改动。"""
    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    # expected_doc_ids = 语料内相对路径(架构 §10.1),不是 sha1 doc_id
    goldens = [
        _golden("g1", ["a.md"]),
        _golden("g2", ["b.md", "a.md"], difficulty="cross_doc"),
        _golden("g3", [], difficulty="out_of_domain"),
    ]
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md, "b.md": md}, goldens=goldens)
    assert run1.exit_code == 0, run1.stdout
    assert "[freshness]" in run1.stdout  # 首次全新增:无变更/删除,真题不受影响
    goldens_bytes = (tmp_path / "goldens.json").read_bytes()

    # 变更 b.md、删除 a.md → g1(期望 a)与 g2(期望 a+b)受影响,g3 域外不受影响
    b_changed = md.replace("发热处理", "发热处置与随访")
    (run1.corpus_dir / "a.md").unlink()
    run2 = run_ingest(
        tmp_path, monkeypatch, capsys, {"b.md": b_changed}, goldens=goldens, argv=("--dry-run",)
    )
    assert run2.exit_code == 0, run2.stdout
    tail = run2.stdout.split("[freshness]", 1)[1]
    assert "g1" in tail and "g2" in tail
    assert "g3" not in tail
    assert "a.md" in tail and "b.md" in tail  # 附源文档路径,供人工反查
    assert "人工" in tail and "永不自动失效" in tail
    assert (tmp_path / "goldens.json").read_bytes() == goldens_bytes

    # 真实 ingest 同样输出保鲜清单
    run3 = run_ingest(tmp_path, monkeypatch, capsys, {"b.md": b_changed}, goldens=goldens)
    assert run3.exit_code == 0, run3.stdout
    assert "g1" in run3.stdout.split("[freshness]", 1)[1]
    assert (tmp_path / "goldens.json").read_bytes() == goldens_bytes


def test_freshness_skips_gracefully_without_goldens(tmp_path, monkeypatch, capsys):
    """真题集缺失:保鲜清单跳过,ingest 不受影响(门禁资产与建索引解耦)。"""
    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md})  # goldens=None → 文件不存在
    assert run1.exit_code == 0, run1.stdout
    run2 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md.replace("发热处理", "发热处置")})
    assert run2.exit_code == 0, run2.stdout  # 有变更但真题集缺失 → 跳过,不挂
    assert "[freshness]" in run2.stdout and "跳过" in run2.stdout


def test_model_dim_mismatch_refuses_query_and_ingest(tmp_path, monkeypatch, capsys):
    """manifest 模型/维度与配置不一致 → 存储层拒查提示 --rebuild;增量 ingest 同样拒绝。"""
    from rag_core.config import Config
    from rag_core.store import IndexMismatchError, open_query_collection

    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md})
    assert run1.exit_code == 0, run1.stdout

    # 配置一致:查询入口放行
    assert open_query_collection(Config.from_env()).count() == 1

    # 维度不一致:拒查,提示 --rebuild
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        open_query_collection(Config.from_env())
    # 模型不一致:同样拒查
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v3")
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        open_query_collection(Config.from_env())

    # 增量 ingest 拒绝混嵌,索引分毫不动
    manifest_bytes = run1.manifest_file.read_bytes()
    ids_before = sorted(run1.collection().get()["ids"])
    run2 = run_ingest(
        tmp_path, monkeypatch, capsys, {"a.md": md},
        env_extra={"EMBEDDING_MODEL": "text-embedding-v3"},
    )
    assert run2.exit_code != 0
    assert "--rebuild" in run2.stdout
    assert run2.manifest_file.read_bytes() == manifest_bytes
    assert sorted(run2.collection().get()["ids"]) == ids_before
    assert not run2.embedder.embedded

    # --rebuild 是出口:换模型后全量重建,manifest 写入新模型/维度
    run3 = run_ingest(
        tmp_path, monkeypatch, capsys, {"a.md": md}, argv=("--rebuild",),
        env_extra={"EMBEDDING_MODEL": "text-embedding-v3"},
    )
    assert run3.exit_code == 0, run3.stdout
    assert run3.manifest()["embedding_model"] == "text-embedding-v3"
    assert open_query_collection(Config.from_env()).count() == 1


def test_query_gate_without_index(tmp_path, monkeypatch, capsys):
    """索引未建:查询入口拒绝并提示先 ingest。"""
    from rag_core.config import Config
    from rag_core.store import IndexMismatchError, open_query_collection

    run_ingest(tmp_path, monkeypatch, capsys, {}, argv=("--dry-run",))
    with pytest.raises(IndexMismatchError, match="ingest"):
        open_query_collection(Config.from_env())


def test_rebuild_wipes_collection_and_manifest(tmp_path, monkeypatch, capsys):
    """--rebuild 清空重建:collection 杂散向量与已删文件一并清掉,manifest 重写。"""
    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md, "b.md": md})
    assert run1.exit_code == 0, run1.stdout
    # 外部注入杂散向量(模拟旧索引残留),rebuild 必须清掉
    run1.collection().add(
        ids=["stray:0"],
        documents=["杂散向量"],
        metadatas=[{"doc_id": "stray"}],
        embeddings=[[0.0] * 1024],
    )
    (run1.corpus_dir / "b.md").unlink()

    run2 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md}, argv=("--rebuild",))
    assert run2.exit_code == 0, run2.stdout
    assert "[rebuild]" in run2.stdout
    assert "added=1 changed=0 deleted=0 skipped=0" in run2.stdout
    col = run2.collection()
    assert sorted(col.get()["ids"]) == [f"{doc_id_of('a.md')}:0"]
    assert list(run2.manifest()["files"]) == ["a.md"]
    assert run2.embedder.embedded  # 全量重嵌


def test_dry_run_reports_three_sets_without_touching_index(tmp_path, monkeypatch, capsys):
    """--dry-run 报告新增/变更/删除三集合,索引与 manifest 分毫不动,零 embedding。"""
    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": md, "b.md": md})
    assert run1.exit_code == 0, run1.stdout
    manifest_bytes = run1.manifest_file.read_bytes()
    ids_before = sorted(run1.collection().get()["ids"])

    a_changed = md + "\n\n### 新增小节\n\n新增内容:复查随访与停药观察的说明文字,足够长,可以独立成为一个检索单元,不触发合并。持续观察体温变化与伴随症状,不要自行加量用药,退烧药需在医生指导下使用。\n"
    (run1.corpus_dir / "b.md").unlink()
    run2 = run_ingest(
        tmp_path, monkeypatch, capsys, {"a.md": a_changed, "c.md": md}, argv=("--dry-run",)
    )
    assert run2.exit_code == 0, run2.stdout
    assert "added=1 changed=1 deleted=1 skipped=0" in run2.stdout
    assert "+ c.md" in run2.stdout and "~ a.md" in run2.stdout and "- b.md" in run2.stdout
    assert "[dry-run]" in run2.stdout
    assert run2.manifest_file.read_bytes() == manifest_bytes
    assert sorted(run2.collection().get()["ids"]) == ids_before
    assert not run2.embedder.embedded


def test_deleted_file_chunks_cleaned(tmp_path, monkeypatch, capsys):
    """删文件清理其 Chunk:按 doc_id 删净,manifest 条目移除,其余文件跳过。"""
    md = "# 文档\n\n## 章节\n\n### 小节\n\n独有内容:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化,不要自行加量用药。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"keep.md": md, "gone.md": md})
    assert run1.exit_code == 0, run1.stdout
    gone_id = doc_id_of("gone.md")
    assert run1.collection().get(where={"doc_id": gone_id})["ids"]

    (run1.corpus_dir / "gone.md").unlink()
    run2 = run_ingest(tmp_path, monkeypatch, capsys, {"keep.md": md})
    assert run2.exit_code == 0, run2.stdout
    assert "added=0 changed=0 deleted=1 skipped=1" in run2.stdout
    assert "- gone.md" in run2.stdout
    assert run2.collection().get(where={"doc_id": gone_id})["ids"] == []
    assert "gone.md" not in run2.manifest()["files"]
    assert "keep.md" in run2.manifest()["files"]
    assert not run2.embedder.embedded  # 删除零 embedding 成本


def test_changed_file_reembedded_alone_unchanged_skipped(tmp_path, monkeypatch, capsys):
    """改一个 md 只重嵌该文件:变更先删后插(旧 chunk 清干净),未变文件零成本跳过。"""
    a_v1 = (
        "# 甲文档\n\n## 章节\n\n### 小节一\n\n甲文档独有内容甲:发热处理与用药间隔的说明文字,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化。成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医。退烧药需在医生指导下使用,两次用药间隔至少4-6小时。\n\n"
        "### 小节二\n\n甲文档第二节:咳嗽观察与就医时机的说明文字,同样足够长,可以独立成为一个检索单元,不触发合并。干咳持续超过2周,或伴随气短、咯血、胸痛,应尽快到呼吸内科检查。湿咳需观察痰色与痰量,黄脓痰可能提示细菌感染,爬楼气短是心肺受累的警示信号,不可忽视。\n"
    )
    b_md = "# 乙文档\n\n## 章节\n\n### 小节\n\n乙文档独有内容乙:挂号指南与科室选择的说明文字,足够长,可以独立成为一个检索单元,不触发合并。发烧、咳嗽等呼吸道症状挂呼吸内科;高血压复诊挂心内科;糖尿病挂内分泌科。症状不明确可先咨询导诊台,就诊请携带既往检查报告与病历。\n"
    run1 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": a_v1, "b.md": b_md})
    assert run1.exit_code == 0, run1.stdout
    b_id = doc_id_of("b.md")
    b_before = run1.collection().get(where={"doc_id": b_id}, include=["documents", "metadatas"])
    b_entry_before = run1.manifest()["files"]["b.md"]
    a_hash_before = run1.manifest()["files"]["a.md"]["content_hash"]

    # 变更 a.md:改写小节一并删掉小节二(chunk 数 2→1,验证先删后插而非仅 upsert)
    a_v2 = "# 甲文档\n\n## 章节\n\n### 小节一\n\n甲文档独有内容甲改写:发热处理与用药间隔说明,新增复查随访与停药观察,足够长,可以独立成为一个检索单元,不触发合并。低烧可先物理降温、多饮水,持续观察体温变化。成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医。退烧药需在医生指导下使用,两次用药间隔至少4-6小时。\n"
    (run1.corpus_dir / "a.md").write_text(a_v2, encoding="utf-8")
    run2 = run_ingest(tmp_path, monkeypatch, capsys, {"a.md": a_v2, "b.md": b_md})
    assert run2.exit_code == 0, run2.stdout
    assert "added=0 changed=1 deleted=0 skipped=1" in run2.stdout
    assert "~ a.md" in run2.stdout

    # 只重嵌变更文件:fake 记录的全部文本来自 a.md,乙文档只字未嵌
    assert run2.embedder.embedded, "变更文件应触发 embedding"
    assert all("甲文档" in t for t in run2.embedder.embedded)
    assert not any("乙文档" in t for t in run2.embedder.embedded)

    # 未变文件:manifest 条目与 Chroma 内容原样
    assert run2.manifest()["files"]["b.md"] == b_entry_before
    b_after = run2.collection().get(where={"doc_id": b_id}, include=["documents", "metadatas"])
    assert b_after["ids"] == b_before["ids"]
    assert b_after["documents"] == b_before["documents"]

    # 变更文件:旧 chunk 先删后插(2→1,多余旧 id 清干净),新文本生效
    a_id = doc_id_of("a.md")
    a_after = run2.collection().get(where={"doc_id": a_id}, include=["documents"])
    assert a_after["ids"] == [f"{a_id}:0"]
    assert "新增复查随访" in a_after["documents"][0]
    assert "咳嗽观察" not in "\n".join(a_after["documents"])
    assert run2.manifest()["files"]["a.md"]["content_hash"] != a_hash_before


def test_h4_and_below_do_not_split(tmp_path, monkeypatch, capsys):
    """H4+ 不切:归入所在 H3 section,标题行留在正文,面包屑止于 H3。"""
    md = (
        "# 指南\n\n## 章节\n\n### 小节\n\n"
        "本小节包含两个 H4 子标题,但 H4 及以下不构成切分边界,整段归入所在 H3 section,只要不超过上限就是一个检索单元。\n\n"
        "#### 甲\n\n内容甲:低烧可先物理降温、多饮水,持续观察体温变化与伴随症状,不要自行加量用药。\n\n"
        "#### 乙\n\n内容乙:成人发烧持续超过3天不退,或伴随呼吸困难、意识改变,应及时就医。\n"
    )
    run = run_ingest(tmp_path, monkeypatch, capsys, {"h4.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    assert len(got["ids"]) == 1
    assert got["metadatas"][0]["heading_path"] == "指南 > 章节 > 小节"
    body = got["documents"][0].split("\n\n", 1)[1]
    assert "#### 甲" in body and "#### 乙" in body
    assert "内容甲" in body and "内容乙" in body


def test_front_matter_metadata_not_in_body(tmp_path, monkeypatch, capsys):
    """front-matter 提 title/tags 为元数据,不进 chunk 正文;无 front-matter 时 title 回退 H1/文件名。"""
    body = "自签收之日起 7 天内可无理由退款,15 天内质量问题换货。退款到账时效:原路退回 1–3 个工作日,节假日顺延。定制商品、已激活的软件授权不支持无理由退款,具体以商品页说明为准。"
    md = f"---\ntitle: 退款政策速查\ntags: [售后, 退款]\n---\n\n# 售后政策\n\n## 退款\n\n### 退款时效\n\n{body}\n"
    md_no_h1 = "## 开票说明\n\n订单完成后在「我的订单」申请开票,电子发票 24 小时内发送至邮箱,纸质发票 3 个工作日内寄出,运费由平台承担,发票抬头须与下单主体一致。\n"
    run = run_ingest(
        tmp_path, monkeypatch, capsys, {"fm.md": md, "no_title.md": md_no_h1}
    )
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    metas = {m["source_path"]: m for m in got["metadatas"]}
    docs = {m["source_path"]: d for d, m in zip(got["documents"], got["metadatas"])}

    assert metas["fm.md"]["title"] == "退款政策速查"  # front-matter 优先于 H1
    assert metas["fm.md"]["tags"] == "售后,退款"  # 逗号拼接标量(Chroma 约束)
    assert "tags:" not in docs["fm.md"]
    assert not docs["fm.md"].startswith("---")
    assert docs["fm.md"].startswith("售后政策 > 退款 > 退款时效\n\n")  # 面包屑仍用标题栈

    assert metas["no_title.md"]["title"] == "no_title"  # 无 H1 → 文件名 stem
    assert metas["no_title.md"]["tags"] == ""


def test_long_section_secondary_split_with_overlap(tmp_path, monkeypatch, capsys):
    """超上限 section 按段落边界二次切分:块 ≤ 上限,相邻块 overlap ≈100 字符,段落不腰斩。"""
    paras = [
        f"第{i:02d}段:用药安全须知,处方药须凭医生处方使用,抗生素不可自行服用,退烧药两次间隔至少4-6小时,24小时不超过4次。"
        for i in range(1, 21)
    ]  # 每段 ≈60 字符,20 段 + 空行 > 1100 上限
    md = "# 用药手册\n\n## 用药安全\n\n### 总则\n\n" + "\n\n".join(paras) + "\n"
    run = run_ingest(tmp_path, monkeypatch, capsys, {"long.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    docs = got["documents"]
    assert len(docs) >= 2
    bodies = [d.split("\n\n", 1)[1] for d in docs]
    for d, body in zip(docs, bodies):
        assert d.startswith("用药手册 > 用药安全 > 总则\n\n")
        assert len(body) <= 1100  # 上限(面包屑头另计)
    # 段落永不腰斩:每段完整落在某个 chunk
    for p in paras:
        assert any(p in b for b in bodies), p
    # overlap ≈100:上一块末尾整段(≤100 字符)出现在下一块开头
    for prev, nxt in zip(bodies, bodies[1:]):
        last_para = prev.rsplit("\n\n", 1)[-1]
        assert len(last_para) <= 100
        assert nxt.startswith(last_para)


def test_short_sections_merge_with_same_parent_sibling(tmp_path, monkeypatch, capsys):
    """过短 section 与相邻同父小节合并(架构附录 A 工作示例);无同父邻居的短 section 独立。"""
    md = (
        "# 售后政策\n## 退款\n### 退款时效\n"
        "自签收之日起 7 天内可无理由退款,15 天内质量问题换货。退款到账时效:原路退回 1–3 个工作日。\n"
        "### 不予退款情形\n定制商品、已激活的软件授权不支持无理由退款。\n"
        "## 发票\n### 开票流程\n订单完成后在「我的订单」申请,电子发票 24 小时内发送至邮箱。\n"
    )
    run = run_ingest(tmp_path, monkeypatch, capsys, {"售后政策.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    assert len(got["ids"]) == 2
    by_path = {m["heading_path"]: d for d, m in zip(got["documents"], got["metadatas"])}

    # 退款时效 + 不予退款情形:同父 H2「退款」,合并;面包屑 = 共同父路径,标题行回注正文
    merged = by_path["售后政策 > 退款"]
    assert merged.startswith("售后政策 > 退款\n\n")
    assert "### 退款时效\n自签收之日起 7 天内可无理由退款" in merged
    assert "### 不予退款情形\n定制商品、已激活的软件授权不支持无理由退款。" in merged

    # 开票流程:无同父邻居,短也独立,保留自身面包屑
    lone = by_path["售后政策 > 发票 > 开票流程"]
    assert lone.startswith("售后政策 > 发票 > 开票流程\n\n订单完成后")


def _big_fence(payload: str) -> str:
    return f"```text\n{payload}\n```"


def _big_table(cell: str) -> str:
    rows = "\n".join(f"| 药品{i} | {cell} |" for i in range(6))
    return f"| 药品 | 说明 |\n|---|---|\n{rows}"


def test_atomic_blocks_never_split_oversize_standalone(tmp_path, monkeypatch, capsys):
    """表格/围栏永不腰斩:超限原子块独立成 Chunk 标 oversize;未超限原子块留在 section 内。"""
    fence_payload = "抗生素用药警示:" + "凭处方使用,不可自行服用。" * 90  # >1100 字符
    table_cell = "须凭医生处方,按疗程足量使用,不可症状缓解即自行停药。" * 8  # 表格整体 >1100
    big_fence = _big_fence(fence_payload)
    big_table = _big_table(table_cell)
    small_fence = "```text\n示例:阿莫西林 0.5g 口服\n```"
    intro = "本节给出完整对照表与示例代码。表格与代码块都是原子块,任何情况下不得腰斩;超限的原子块独立成为检索单元并标记 oversize,由生成端在检索后取舍。这段引言足够长,可以独立成为一个检索单元。"
    md = (
        "# 用药手册\n\n## 抗生素说明\n\n"
        f"### 处方原则\n\n抗生素不可自行服用,须凭医生处方,按疗程足量使用。本节说明处方原则与示例,内容足够长,可以独立成为一个检索单元,不需要与相邻小节合并。孕妇、儿童、肝肾功能不全者用药须遵医嘱,不可自行调整剂量。退烧药两次间隔至少4-6小时,24小时不超过4次。\n\n{small_fence}\n\n"
        f"### 对照表与示例\n\n{intro}\n\n{big_fence}\n\n{big_table}\n"
    )
    run = run_ingest(tmp_path, monkeypatch, capsys, {"atomic.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    docs = got["documents"]
    metas = got["metadatas"]

    # 大围栏:完整落在恰好一个 chunk,标 oversize
    fence_hits = [i for i, d in enumerate(docs) if fence_payload in d]
    assert len(fence_hits) == 1
    assert big_fence in docs[fence_hits[0]]
    assert metas[fence_hits[0]]["oversize"] is True

    # 大表格:完整落在恰好一个 chunk,标 oversize
    table_hits = [i for i, d in enumerate(docs) if table_cell in d]
    assert len(table_hits) == 1
    assert big_table in docs[table_hits[0]]
    assert metas[table_hits[0]]["oversize"] is True

    # 未超限原子块与正文同 chunk,不标 oversize
    small_hits = [i for i, d in enumerate(docs) if small_fence in d]
    assert len(small_hits) == 1
    assert "抗生素不可自行服用" in docs[small_hits[0]]
    assert metas[small_hits[0]]["oversize"] is False

    # 超限原子块独立成 chunk:该 chunk 除面包屑外只有原子块本身
    assert docs[fence_hits[0]].split("\n\n", 1)[1] == big_fence
    assert docs[table_hits[0]].split("\n\n", 1)[1] == big_table


def test_ingest_basic_doc_end_to_end(tmp_path, monkeypatch, capsys):
    """一条命令把语料建成索引:面包屑头 + metadata schema + chunk id 规则 + manifest。"""
    run = run_ingest(tmp_path, monkeypatch, capsys, {"basic.md": BASIC_MD})
    assert run.exit_code == 0, run.stdout

    col = run.collection()
    assert col.count() == 1
    got = col.get(include=["documents", "metadatas", "embeddings"])
    doc_id = doc_id_of("basic.md")
    assert got["ids"] == [f"{doc_id}:0"]

    # 面包屑注入 chunk 头部,正文原样(标题行不进正文,面包屑承载)
    text = got["documents"][0]
    assert text.startswith("就诊指南 > 发烧处理 > 体温判定\n\n")
    assert "腋下体温≥37.3℃即属发烧" in text
    assert "# 就诊指南" not in text

    meta = got["metadatas"][0]
    assert meta["doc_id"] == doc_id
    assert meta["source_path"] == "basic.md"
    assert meta["title"] == "就诊指南"
    assert meta["heading_path"] == "就诊指南 > 发烧处理 > 体温判定"
    assert meta["chunk_index"] == 0
    assert meta["content_hash"]
    assert meta["tags"] == ""
    assert meta["oversize"] is False

    # 向量维度 = 配置维度(fake 确定性,零网络)
    assert len(got["embeddings"][0]) == 1024

    manifest = run.manifest()
    assert manifest is not None
    assert manifest["embedding_model"] == "text-embedding-v4"
    assert manifest["embedding_dim"] == 1024
    assert manifest["index_version"] == 1
    entry = manifest["files"]["basic.md"]
    assert entry["doc_id"] == doc_id
    assert entry["chunk_ids"] == [f"{doc_id}:0"]
    assert entry["content_hash"]
    assert entry["mtime"] > 0


def test_module_entrypoint_via_env_backend(tmp_path, monkeypatch):
    """python -m rag_core.ingest 真实入口:env 分发 fake 后端,零注入零网络。

    覆盖 make_embedder 的 env 分发路径(注入测试不经过它)。
    """
    import os
    import subprocess
    import sys

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "entry.md").write_text(BASIC_MD, encoding="utf-8")
    env = {
        **os.environ,
        "RAG_CORPUS_DIR": str(corpus),
        "RAG_INDEX_DIR": str(tmp_path / "index"),
        "RAG_MANIFEST_FILE": str(tmp_path / "manifest.json"),
        "RAG_GOLDENS_FILE": str(tmp_path / "goldens.json"),
        "EMBEDDING_BACKEND": "fake",
        "EMBEDDING_MODEL": "text-embedding-v4",
        "EMBEDDING_DIM": "1024",
    }
    env.pop("DASHSCOPE_API_KEY", None)  # hermetic:即使 backend 配错也无 key 可用

    proc = subprocess.run(
        [sys.executable, "-m", "rag_core.ingest"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "added=1" in proc.stdout
    assert (tmp_path / "manifest.json").exists()

    dry = subprocess.run(
        [sys.executable, "-m", "rag_core.ingest", "--dry-run"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "added=0 changed=0 deleted=0 skipped=1" in dry.stdout


def test_merge_with_oversize_sibling_keeps_atomic_standalone(tmp_path, monkeypatch, capsys):
    """短小节并入含超限围栏的同父兄弟:围栏仍独立成 Chunk 标 oversize,合并不得吞原子块。"""
    fence_payload = "合并场景警示:" + "凭处方使用,不可自行服用。" * 90  # >1100
    big_fence = _big_fence(fence_payload)
    long_body = (
        "抗生素须凭医生处方,按疗程足量使用,不可症状缓解即自行停药。"
        "孕妇、儿童、肝肾功能不全者用药须遵医嘱,不可自行调整剂量。"
        "退烧药两次间隔至少4-6小时,24小时不超过4次,连续使用不超过3天。"
    )
    md = (
        "# 用药手册\n\n## 抗生素\n\n"
        f"### 短提示\n\n简短提示:详见下文完整说明。\n\n"
        f"### 完整说明\n\n{long_body}\n\n{big_fence}\n"
    )
    run = run_ingest(tmp_path, monkeypatch, capsys, {"merge_atomic.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    docs, metas = got["documents"], got["metadatas"]

    # 超限围栏:完整落在恰好一个 chunk,独立、标 oversize
    hits = [i for i, d in enumerate(docs) if fence_payload in d]
    assert len(hits) == 1
    assert docs[hits[0]].split("\n\n", 1)[1] == big_fence
    assert metas[hits[0]]["oversize"] is True

    # 其余 chunk 均不超上限(面包屑除外按正文计),短提示内容不丢
    for d, m in zip(docs, metas):
        body = d.split("\n\n", 1)[1]
        assert m["oversize"] or len(body) <= 1100, (m["heading_path"], len(body))
    assert any("简短提示:详见下文完整说明。" in d for d in docs)
    assert any("抗生素须凭医生处方" in d for d in docs)


def test_chained_short_siblings_never_exceed_max_unflagged(tmp_path, monkeypatch, capsys):
    """连续多个短小节合并后超上限:仍须二次切分,不得产出未标记的超限 chunk。"""
    shorts = "".join(
        f"### 情形{i}\n\n情形{i}的处理说明:先观察精神状态,多饮水,物理降温;持续不退应及时就医,用药遵医嘱,不可自行加量。"
        f"两次用药间隔至少4-6小时,24小时不超过4次。儿童与孕妇用药须遵医嘱,肝肾功能不全者须调整剂量。\n\n"
        for i in range(12)
    )
    md = f"# 手册\n\n## 发热\n\n{shorts}"
    run = run_ingest(tmp_path, monkeypatch, capsys, {"chain.md": md})
    assert run.exit_code == 0, run.stdout

    got = run.collection().get(include=["documents", "metadatas"])
    assert len(got["ids"]) > 1  # 12 短节 ≈1000+ 字符,不得塞进一个 chunk
    for d, m in zip(got["documents"], got["metadatas"]):
        body = d.split("\n\n", 1)[1]
        assert m["oversize"] or len(body) <= 1100
    for i in range(12):
        assert any(f"情形{i}的处理说明" in d for d in got["documents"])
