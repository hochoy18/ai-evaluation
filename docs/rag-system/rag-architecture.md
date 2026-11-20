# RAG 系统架构设计 —— 个人 Markdown 知识库

> 本文定位:**架构设计文档,不是实现**。所有代码块均为契约声明或伪代码,落地按第 13 章路线分阶段执行。
> 姊妹文档:[RAG 评估学习计划](../evaluation-frameworks/deepeval-rag-evaluation-plan.md)(评测线四步打法)、[评测框架选型调研](../evaluation-frameworks/README.md)(指标理论)。
> 语料落定:本仓库 `docs/**/*.md`(13 篇,AI 评测知识库),**Markdown/纯文本为主**,个人/小团队知识库规模;架构按百到千篇量级设计,真实业务文档到位后换目录即可。
> 技术栈基线:uv + Python 3.12,阿里云百炼 OpenAI 兼容接口(qwen 生成),DeepEval >= 4.3.4 + Ragas >= 0.4.4。
> 设计日期:2026-11-19。文中所有数值参数(chunk 大小、阈值、top-k)均为**初值,以 golden set 评测校准**。
> 修订:2026-11-19 grilling 共识回写(18 项决策;key/embedding/rerank 实测记录见附录 E)。
> 领域文档:术语规范见根目录 `CONTEXT.md`;解耦与低置信决策立 ADR 0001/0002(`docs/adr/`)。

---

## 1. 背景与目标

### 1.1 现状

- 仓库已有一个**关键词版最小 RAG**:`src/deepeval_cases/rag_app.py`(146 行),是 RAG 评测学习计划的第 1 步产出:
  - 知识库:6 篇内嵌中文医疗短文档(带 keywords 列表);
  - 检索:`retrieve(question, k=3)` 关键词命中计数打分,返回**有序 `list[str]`**;
  - 生成:`generate(question, contexts)` 调百炼 qwen,`temperature=0.0`,上下文以 `[资料N]` 编号拼 prompt;
  - 插桩:`@observe(type="agent"/"retriever"/"llm")` 三层 span 树;
  - 失败钩子:`--bad-retrieval`(检索端污染)/`--hallucinate`(生成端污染)。
- 评测侧 `test_rag.py` **尚未编写**(学习计划第 2 步),四指标门禁还没建立。
- 仓库无任何向量库/embedding 代码;依赖里没有 chromadb/faiss/langchain。
- 语料落定(grilling):无外部业务文档,语料 = 本仓库 `docs/**/*.md`(13 篇),域 = AI 评测知识库,非涉密;若未来有真实业务文档:换 `RAG_CORPUS_DIR` + 重出题 + `--rebuild`,架构不动。
- 实测(附录 E):token-plan key 对 embedding/rerank **不可用**(404/401);已另备按量计费 `DASHSCOPE_API_KEY`(普通 sk- key),embedding batch=10 与 gte-rerank-v2 均实测通过。

### 1.2 目标

把关键词版升级为一个面向 **Markdown 业务文档**的完整 RAG 系统:

1. ** ingestion 流水线**:业务 md 文档 → 结构感知切分 → embedding → 本地向量索引,支持增量更新;
2. **检索端**:dense 向量检索起步,按需升级 BM25 混合 + rerank 精排,每一步升级由评测数据决定去留;
3. **生成端**:引用溯源、软拒答,沿用现有 prompt 骨架;
4. **评测与可观测**:四指标门禁 + 非 LLM 检索指标 + 失败注入 + 本地 tracing,贯穿所有阶段。

### 1.3 第一原则(不可妥协)

> **评估体系与被测系统解耦:升级检索端,评估代码一行不改。**

这是学习计划的核心知识点,本设计把它做成机制(第 9 章接口契约 + `RAG_RETRIEVER` 后端开关),而不是口号。推论:

- **评测先行**:先在关键词版上建好四指标基线(P0),再动检索端——没有基线指纹,「一行不改」与「升级有效」都无从证明;
- 检索内部随便重构,门面三签名与 `retrieval_context: list[str]` 有序契约冻结;
- 组件选型争议(要不要 BM25?要不要 rerank?)不靠辩论,靠同一 golden set 上的配置对比矩阵(第 10 章)。

---

## 2. 范围界定

### In(本设计覆盖)

| 能力 | 说明 |
|---|---|
| Markdown/纯文本 ingestion | 解析、标题感知切分、元数据、增量更新 CLI |
| 嵌入式向量检索 | 百炼 embedding + Chroma 本地持久化 |
| 混合检索 + 精排 | BM25(jieba)+ RRF 融合 + gte-rerank,分阶段启用 |
| 引用与拒答 | `[资料N]` 溯源映射、软拒答 + 低置信标记 |
| 评测 | golden set × 四指标 + hit-rate/MRR + 失败注入 + 在线抽样 |
| 可观测 | DeepEval `@observe` 本地 tracing(`deepeval inspect`) |

### Out(逐条裁剪理由)

| 不做 | 理由 |
|---|---|
| PDF/Word/多模态/OCR 解析 | 语料以 Markdown 为主;版面解析是独立大课题,出现真实需求再立项 |
| 权限与多租户 | 个人/小团队知识库,无隔离需求 |
| 分布式、服务化 API(FastAPI 等) | 进程内嵌入式足够;服务化只增加运维面,不增加检索质量 |
| query 改写 / 多跳 / agentic RAG / GraphRAG | 先把单跳检索质量做扎实;这些是检索端达标后的进阶课题 |
| 多轮对话记忆 | 评测四指标是单轮指标,多轮会引入 KnowledgeRetention 等新维度,分散主线 |
| 模型微调 | 数据量与收益不匹配 |
| 实时文件监听(watchdog) | 个人库手动跑 ingestion CLI 足够 |
| Langfuse 等外部观测栈 | 仓库尚无集成;本地 inspect 够用,列为后续演进(学习计划阶段 5) |

---

## 3. 总体架构与数据流

### 3.1 双流水线

**离线建索引**(独立 CLI,手动触发,不在查询链路上):

```
docs/**/*.md(RAG_CORPUS_DIR,语料 = 本仓库 docs/)
  → 扫描 + 对照 manifest(content hash)→ 新增 / 变更 / 删除 三个集合
  → 解析:markdown-it-py token 流(标题层级、front-matter、表格/代码块边界)
  → 切分:标题感知 section 化 → 超长二次切分 → 面包屑注入 → Chunk 列表
  → 元数据:doc_id / source_path / title / heading_path / chunk_index / content_hash …
  → embedding:text-embedding-v4 批量调用(批量上限 + 重试退避)
  → Chroma upsert(变更文件按 doc_id 先删后插;删除文件按 doc_id 清理)
  → 写回 manifest(embedding 模型名 + 维度 + 索引版本)
```

**在线查询**(span 树即埋点图):

```
rag_app (agent span)                     ← update_current_trace(input, output, retrieval_context)
├── retrieve (retriever span)            ← 契约不变:question, k=3 → 有序 list[str]
│   │                                       出口 update_current_span(retrieval_context=top-k 文本)
│   │                                       + span 属性:N→source 映射 / 低置信标记 / 后端名
│   │  [BAD_RETRIEVAL 钩子在此层最前,先于后端分发 —— 三种后端共享同一注入点]
│   ├── embed_query (tool span)          ← input=question, output=维度/耗时
│   ├── vector_search (tool span)        ← Chroma top-20, output=候选 id+score
│   ├── bm25_search (tool span)          ← hybrid 模式才有;jieba 分词,内存索引 top-20
│   ├── rrf_fuse (tool span)             ← RRF(k=60) 融合取 10, output=融合序
│   └── rerank (tool span)               ← gte-rerank-v2, 10→3;不可用时 Identity 直通并标记降级
└── generate (llm span)                  ← 契约不变:[资料N] 拼 prompt, temp=0.0
                                            [HALLUCINATE 钩子在此,prompt 翻转,原样保留]
```

要点:**外层三个 span 的类型、位置、`update_current_*` 调用与现版 `rag_app.py` 完全一致**。新增子 span 只是把 retriever 内部打开——`deepeval inspect` 里树变深,四指标读的数据字段不变。keyword 后端下 retrieve 内部无子 span,树形与今天相同。

### 3.2 模块划分

```
src/
├── rag_core/                    # 新增:RAG 引擎(本设计的实现主体,落地时创建)
│   ├── config.py                # 全部参数集中于此(env 覆盖),不硬编码
│   ├── chunking.py              # Markdown 解析 + 结构感知切分
│   ├── embeddings.py            # Embedder 协议 + 百炼实现(本地备选留位)
│   ├── store.py                 # Chroma 封装 + manifest 读写
│   ├── retrievers.py            # Retriever 协议 + keyword/vector/hybrid 三实现
│   ├── rerank.py                # gte-rerank 客户端 + Identity 降级
│   ├── ingest.py                # 建索引 CLI(python -m rag_core.ingest)
│   └── online_sample.py         # 在线抽样 CLI(python -m rag_core.online_sample)
├── deepeval_cases/
│   ├── rag_app.py               # 门面:三签名冻结,env 分发后端,失败钩子保留
│   └── test_rag.py              # 评测门禁(P0 产出,只 import rag_app,不感知后端)
└── ai_evaluation/models.py      # bailian_judge() 裁判适配(已有,评测复用)
```

### 3.3 数据目录布局

```
docs/                            # 语料 = 本仓库 docs/(13 篇 md,RAG_CORPUS_DIR=./docs,随仓库进 git)
data/
├── rag_index/                   # Chroma 持久化目录(gitignore)
├── rag_manifest.json            # 增量更新清单(gitignore)
├── rag_goldens.json             # 真题 golden set(进 git,评测资产)
├── rag_goldens_medical.json     # 医疗 P0 golden(合成内容,进 git)
├── rag_query_log.jsonl          # 在线查询日志(门面追加写,gitignore)
└── rag_online_report_<date>.md  # 在线抽样报告(gitignore)

不另建 rag_corpus/ 目录;真实业务文档到位后换 RAG_CORPUS_DIR 即可。
```

---

## 4. Ingestion 设计

### 4.1 解析器选型

| 候选 | 结论 | 理由 |
|---|---|---|
| **markdown-it-py** | ✅ 选用 | token 流完整保留标题层级、表格/代码围栏边界、front-matter 可扩展;CommonMark 兼容;纯 Python 零运维 |
| mistune | 备选 | 更快,但 AST 对表格/围栏的结构化程度略弱 |
| 手写正则 | ❌ | 嵌套列表、围栏代码块内的 `#` 等边界情况必然踩坑 |

### 4.2 结构感知切分算法

规则(伪代码级):

```
1. 解析 token 流,维护标题栈 [H1, H2, H3];
   遇到 H1–H3 → 关闭当前 section,开新 section,记录 heading_path(面包屑)。
   H4 及以下不切,归入所在 H3 section。
2. 表格、围栏代码块作为原子块:整体归属当前 section,永不腰斩;
   单块超上限时独立成 chunk 并标记 oversize=true(不切断,检索后由生成端取舍)。
3. section → chunk(len 以字符计,不引 tokenizer):
   - len < 下限(140 字符)→ 与相邻同父小节合并;
   - 下限 ≤ len ≤ 上限(1100 字符)→ 整段成 chunk;
   - len > 上限 → 按段落边界二次切分,目标 ≈700 字符,相邻块 overlap ≈100 字符。
4. 每个 chunk 头部注入面包屑:"{H1} > {H2} > {H3}\n\n{正文}"
   —— 保留「哪个产品/哪个流程下的规定」上下文,也是引用溯源的载体。
5. front-matter(若有)提取为元数据(tags/title),不进 chunk 正文。
```

### 4.3 chunk 参数表(初值,评测校准;以字符计量)

| 参数 | 初值 | 说明 |
|---|---|---|
| 目标大小 | ≈700 字符(≈500 token) | 匹配「一个小节讲清一件事」的粒度 |
| 上限 | 1100 字符(≈800 token) | 超过则段落二次切分 |
| 下限 | 140 字符(≈100 token) | 过短 section 与相邻小节合并,防碎片 |
| 二次切分 overlap | ≈100 字符(10–15%) | 仅 section 内二次切分使用;section 之间不 overlap(标题边界即语义边界) |

计量方式:`len(text)` 字符数代理——确定性、零依赖、扫参可复现;不引 tokenizer(tiktoken 与 qwen 词表不一致),token 换算仅作直觉说明,近似误差由 golden set 校准吸收。

校准方式:在 golden set 上扫 2–3 组参数,看 ContextualPrecision/Recall 与 hit-rate@3(第 10 章),不凭感觉定。

### 4.4 metadata schema

| 字段 | 类型 | 用途 |
|---|---|---|
| doc_id | str | 相对路径稳定 hash(如 sha1[:12]),增量删改的键 |
| source_path | str | 语料内相对路径,溯源反查 |
| title | str | 文档 H1 或 front-matter title |
| heading_path | str | 面包屑,如 `售后政策 > 退款 > 退款时效` |
| chunk_index | int | 文档内全局序号(id 唯一性要求,跨 section 连续);chunk id = `{doc_id}:{chunk_index}` |
| content_hash | str | chunk 文本 hash,调试用 |
| tags | str | front-matter tags 逗号拼接(Chroma metadata 值须为标量) |
| oversize | bool | 超限标记:原子块超限独立成 chunk,或正文超上限且无法按块边界再切(如单个超长段落);不存在未标记的超限 chunk |

### 4.5 增量更新状态机

manifest(`data/rag_manifest.json`)记录:`embedding_model`、`embedding_dim`、`index_version`、`files: {相对路径: {content_hash, doc_id, chunk_ids, mtime}}`。

```
扫描语料目录,对每个文件:
├── 不在 manifest            → 新增:切分 + embed + upsert
├── 在 manifest,hash 变了    → 变更:按 doc_id delete 旧 chunks → 重新切分 embed upsert
├── 在 manifest,hash 未变    → 跳过(零成本)
└── manifest 有,文件没了     → 删除:按 doc_id delete

启动查询前校验:manifest.embedding_model/dim 与当前配置不一致
→ 拒绝查询,提示 `--rebuild`(防「换模型后新旧向量混检索」的静默错误)
```

千篇量级全量重建也只要分钟级,增量机制的价值是**便宜 + 防模型混用**,不是性能刚需。

### 4.6 CLI 契约

```bash
uv run python -m rag_core.ingest            # 增量:按 manifest 差量更新
uv run python -m rag_core.ingest --rebuild  # 全量:清空 collection 重建
uv run python -m rag_core.ingest --dry-run  # 只报告 新增/变更/删除 集合,不动索引
```

**真题保鲜**:ingest(含 --dry-run)结束时输出「受影响真题」清单——变更/删除的源文档 ∩ 各真题 expected_doc_ids,人工 review;不自动失效、不静默剔出门禁(门禁构成永不静默变化)。

---

## 5. Embedding 设计

### 5.1 选型对比

| 维度 | **百炼 text-embedding-v4(选用)** | 本地 BGE-M3(备选) |
|---|---|---|
| 接入 | 按量计费 `DASHSCOPE_API_KEY`(普通 sk- key)+ `dashscope.aliyuncs.com`;**token-plan key 实测不可用**(附录 E) | sentence-transformers/FlagEmbedding,下载约 2GB 模型 |
| 中文效果 | 强,业务术语覆盖好 | 强,多语言 |
| 延迟 | 网络调用,批量下可接受 | CPU 推理慢,GPU 则增加运维 |
| 成本 | 0.5 元/百万 token;新人免费额度 100 万 token(90 天,北京);docs/ 语料索引远低于此 | 电费 |
| 隐私 | 语料出网(已落定非涉密) | 完全离线 ✅ |

**决策:默认百炼 v4(按量计费 key)**。⚠️ 原「复用现有 OPENAI_BASE_URL/OPENAI_API_KEY、零新增账号」前提已被实测推翻:Token Plan 套餐不覆盖任何 embedding/rerank 模型(官方文档 + 三路探针,附录 E)。**双 key 双 endpoint**:生成端留 token-plan(`OPENAI_API_KEY`),embedding/rerank 走按量计费(`DASHSCOPE_API_KEY`)。BGE-M3 定位 = **零计费退路**(完全不想碰按量计费时)或语料涉密时的选择——`Embedder` 协议(第 9 章)保留切换位,索引需 `--rebuild`(模型名写入 manifest 强制校验)。

### 5.2 关键参数与工程细节

- **维度 1024**(v4 默认,`dimensions` 参数可调,实测确认);维度写入 manifest,查询时校验一致性;
- **批量上限 10 条**(v3/v4 一致,单条 8192 token):官方文档确认 + **实测确认**(batch=11 返回 400「should not be larger than 10」,batch=10 成功,附录 E);`EMBEDDING_BATCH=10` 写 config;
- 失败重试 + 限流退避(指数退避,429 单独处理);
- 查询端与索引端**必须同模型同维度**,由 manifest 校验兜底;
- 成本估算写入 ingestion 日志(输入 token 数),可观测。

---

## 6. 向量存储

### 6.1 选型对比

| 候选 | 结论 | 理由 |
|---|---|---|
| **Chroma(嵌入式)** | ✅ 选用 | `PersistentClient` 进程内嵌入、纯 pip 安装零运维;原生 metadata 过滤、按 doc_id 批量 delete/upsert(增量刚需)、cosine HNSW;万级 chunk 毫无压力 |
| LanceDB | 最强备选 | 同为嵌入式,**内建 FTS**——若混合检索想省掉自建 BM25 可联动换库;生态略新 |
| FAISS | ❌ | 只有索引内核,metadata/持久化/增删全要手搓 |
| Qdrant | ❌(本规模) | server 模式对个人库过重;local 模式生态不如 Chroma 顺手 |

### 6.2 Collection schema

```python
client = chromadb.PersistentClient(path="data/rag_index/")
col = client.get_or_create_collection(
    name="rag_corpus",
    metadata={"hnsw:space": "cosine"},
)
# id 规则:{doc_id}:{chunk_index}
# documents=chunk 文本(含面包屑头), metadatas=4.4 节 schema, embeddings=1024 维
```

- 持久化目录 gitignore;损坏/换模型 → `--rebuild` 全量重建,分钟级,不做备份策略;
- **何时才需要换库**:chunk 数到百万级、需要多进程并发写、需要服务化——本规模明确永远不需要,写出来是为了防止过度设计。

---

## 7. 检索设计

### 7.1 top-k 漏斗(初值)

| 阶段 | 数量 | 说明 |
|---|---|---|
| dense 召回 | 20 | Chroma cosine top-20 |
| BM25 召回(hybrid) | 20 | jieba 分词,内存索引 |
| RRF 融合 | 10 | k=60 |
| rerank 精排 | **3** | 与现签名默认 `k=3` 一致,`RAG_TOP_K` 可覆盖 |

前宽后窄:召回阶段保查全,末端窄保 prompt 精简。对外契约 k=3 不变。

### 7.2 RRF 融合

```
score(d) = Σ_over_routes  1 / (k + rank_route(d)),  k = 60
```

RRF 只用排名不用分数,天然规避「cosine 相似度与 BM25 分数量纲不可比」问题。计算例见附录 B。

### 7.3 BM25 路

- `rank_bm25.BM25Okapi` + **jieba 分词**;领域专名(产品名、错误码、制度编号)用 `jieba.load_userdict` 自定义词典维护;
- 索引构建:启动时从 Chroma 全量拉 chunk 文本内存构建——千篇量级秒级完成,**不需要第二个存储**;
- 存在价值:业务文档充满精确术语,dense 对低频专名召回弱,BM25 恰好补。**是否保留由 P2 评测数据决定**。

### 7.4 Rerank 精排

- 模型:百炼 **`gte-rerank-v2`**,候选 10 → 精排取 top-3;备选 qwen3-rerank(支持 OpenAI 兼容接口,记录在案);
- 接口:DashScope 原生 HTTP(官方确认 gte-rerank-v2 **不支持** OpenAI 兼容接口);**用按量计费 `DASHSCOPE_API_KEY`,非 token-plan key**(后者实测 404/401,附录 E);按量 key 下已实测调通;
- 官方限额:最大 500 文档、单条 4000 token、请求总输入 30000 token(query×文档数 + 文档 token 总和)——漏斗 10×700 字符远低于限;
- **分数语义(官方):`relevance_score` ∈ [0,1] 为当前请求内的相对分数,不可作为跨请求比较的绝对值**。实测佐证:强相关文档 top-1 也仅 ≈0.19(附录 E)→ 本设计**不在其上做固定阈值**,只用于排序,原始分记 span 供诊断;低置信改用 cosine(7.5);
- 计费:0.8 元/百万 token,**无免费额度**(2026-10-29 下架);实测单次 3 候选 63 token,成本可忽略;
- 降级路径:rerank 不可用(超时/报错)→ **Identity 直通**(按 RRF 序取前 k),span 标记 `rerank_degraded=true`,系统不因此挂掉;
- 去留标准:P2 对比矩阵中 ContextualPrecision 无显著提升 → 数据驱动砍掉,并记录结论(这本身就是评测解耦价值的演示)。

### 7.5 低置信与拒答联动

- 低置信信号 = **dense 路 cosine top-1 相似度**(绝对几何量,跨请求可比):top-1 相似度 < 阈值(初值 **0.35**,P3 用 out_of_domain 题集分布校准)→ 标记 `low_confidence=true` 写入 retriever span 属性;vector/hybrid 共用 `RAG_LOW_CONFIDENCE`;
- 实现注意:Chroma 返回 cosine **distance**,相似度 = 1 − distance,勿直接拿 distance 比阈值;
- **为什么不用 rerank 分**:官方明言请求内相对分数(7.4),固定阈值无官方依据,实测强相关也仅 ≈0.19;rerank 只排序,原始分记 span;
- keyword 后端无分数,不标记;P3 只在最终选定后端上校准一套阈值;
- **软拒答(默认)**:不改变 retrieve 返回,照常生成,靠 prompt「资料不足明确说明」兜底;低置信标记供在线监控统计;
- **硬拒答(实验开关,不默认)**:cosine 低于阈值返回空列表短路生成——省一次 LLM 调用,但会掩盖「检索差但生成尽责」的诊断场景(第 10 章指纹表),仅做实验用。

### 7.6 三后端行为矩阵

| `RAG_RETRIEVER` | 召回 | 融合 | 精排 | 子 span |
|---|---|---|---|---|
| `keyword`(默认) | 医疗 P0:keywords 命中计数(原样);语料模式:字符 bigram 重叠计数 | — | — | 无(树形与今天相同) |
| `vector` | dense 20 | — | 可选 | embed_query / vector_search (/ rerank) |
| `hybrid` | dense 20 + BM25 20 | RRF→10 | gte-rerank→3 | 全量 |

keyword 语料模式:问题切二字组,数在 chunk 文本(含面包屑)中的命中——确定性、零新依赖、**故意弱**(基线的价值是诚实地弱);在内存走同一套 chunking,不碰 Chroma/embedding;jieba 严格留 P2 hybrid,「vector 相对 keyword 的增益」量的是 dense 语义能力,不是分词器差异。

默认 keyword 保证现有基线与失败注入实验**可复现**;升级是显式选择,不是静默替换。

---

## 8. 生成设计

### 8.1 prompt 模板

沿用现版 `SYSTEM_BASE` 骨架,补引用格式约束。角色随语料域替换:医疗 P0 保持「医疗咨询助手」;语料模式(docs/)用「**AI 评测知识库助手**」:

```
你是{业务域}助手。只依据下方参考资料回答:资料不足时明确说明「资料中没有相关信息」,不要编造。
回答简洁,引用资料编号(如[资料1]);同一结论有多条依据时并列编号。
```

- user prompt 结构不变:`参考资料:\n[资料1] …\n\n用户问题:{question}`;
- chunk 文本自带面包屑头,LLM 看得到「这段出自哪个文档哪个小节」,引用更有的放矢;
- `temperature=0.0`、`APP_MODEL` env 覆盖(默认 qwen4-max-preview)沿用;
- `SYSTEM_HALLUCINATE` 失败注入 prompt 原样保留。

### 8.2 引用溯源映射

- 引擎内部返回 `list[Chunk]`(text + score + metadata);**门面剥成 `list[str]`**,契约不变;
- `N → (source_path, heading_path)` 映射写入 **retriever span 属性**,不进返回字符串;
- **为什么不把来源路径拼进 context 字符串**:judge 会把路径当作内容参与 claims 判定,污染 Faithfulness/ContextualPrecision;溯源信息放 trace 里,答案中 `[资料N]` 照样可反查到具体文件与小节;
- 反查验证(P3 验收):任取答案中一个 `[资料N]`,能在 inspect 的 span 属性里找到对应文件+小节。

---

## 9. 接口契约与平滑升级

### 9.1 冻结声明(评估侧零改动的前提)

```python
# src/deepeval_cases/rag_app.py —— 门面,以下三签名与装饰器位置冻结
@observe(type="retriever")
def retrieve(question: str, k: int = 3) -> list[str]: ...   # 有序!排名即数据

@observe(type="llm")
def generate(question: str, contexts: list[str]) -> str: ...

@observe(type="agent")
def rag_app(question: str) -> tuple[str, list[str]]: ...    # (答案, 检索上下文)
```

- `update_current_span(retrieval_context=top)` 与 `update_current_trace(input, output, retrieval_context)` 位置不变;
- `BAD_RETRIEVAL` 钩子保持在 retrieve 最前(先于后端分发),`HALLUCINATE` 保持在 generate 的 prompt 选择处——三后端共享同一注入点;
- `retrieval_context` 是**按排名有序的 `list[str]`**,不是拼接字符串——ContextualPrecision 按序加权,这是契约里最容易被破坏的一条;
- 门面 `rag_app` 末尾追加写 `data/rag_query_log.jsonl`(question / answer / retrieval_context / low_confidence / backend / timestamp),**写入失败静默降级**(日志不能挂掉查询链路);签名与 span 数据不动,契约不破——这是 10.7 在线抽样的数据源。

### 9.2 内部协议(rag_core 内部使用,不外泄到评测侧)

```python
@dataclass
class Chunk:
    text: str                  # 含面包屑头
    score: float
    metadata: dict             # 4.4 节 schema

class Retriever(Protocol):
    def search(self, question: str, k: int) -> list[Chunk]: ...

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

门面处用 env 分发后端(不改 import 侧),rag_core 内部用协议(可测试、可替换)——两层兼得。

### 9.3 「评估侧零 diff」的证明方式

同一份 `test_rag.py`(只 import `rag_app`)在 `RAG_RETRIEVER=keyword/vector/hybrid` 下各跑一次:

```bash
RAG_RETRIEVER=keyword uv run deepeval test run src/deepeval_cases/test_rag.py
RAG_RETRIEVER=vector  uv run deepeval test run src/deepeval_cases/test_rag.py
RAG_RETRIEVER=hybrid  uv run deepeval test run src/deepeval_cases/test_rag.py
```

test_rag.py 的 git diff 为空 + 分数矩阵产出 = 解耦原则的实证。

**运行器正典 = `deepeval test run`**:本身即 pytest 包装器(非自写循环的「脚本形态」),产出 inspect 排障所需的 test_run 快照;裸 pytest 可跑 assert_test 作快速调试,但无产物链。学习计划「直接 pytest 形态」与此一致(reconciliation 已注记)。

---

## 10. 评测设计

### 10.1 golden set

- 构成:**30 条 = 18 single_hop + 6 cross_doc + 6 out_of_domain**(门禁集 24 条,拒答集 6 条);
- 出题(语料 = 本仓库 docs/,域 = AI 评测知识库):agent 起草 30 条候选,**用户审核并用自己的措辞重写至少一半**——真题的「真」在于用户会这么问,全由 agent 出题会让评测变成「我的理解测我的理解」;Ragas testset generation 仅 P4 扩充 single_hop,标 `synthetic`,永不替代真题;
- 医疗 P0 另备 ~10 条(7 single_hop + 1–2 cross_doc + 1–2 out_of_domain,agent 起草用户审),存 `data/rag_goldens_medical.json`(合成内容,进 git);
- schema:

```json
{
  "id": "g001",
  "question": "……",
  "expected_output": "标准答案(供 ContextualPrecision/Recall)",
  "expected_doc_ids": ["evaluation-frameworks/README.md"],
  "difficulty": "single_hop | cross_doc | out_of_domain",
  "source": "human | synthetic",
  "notes": "出题依据"
}
```

- `expected_doc_ids` = 语料内相对路径,让「该捞哪篇」有**确定性判据**,支撑非 LLM 指标;
- **schema 校验脚本前置**:cross_doc 必须 ≥2 个 expected_doc_ids;out_of_domain 必须为空;格式错误在跑指标之前报错,而不是让 judge 静默给 0 分;
- **纯域出题规则**:真题必须纯域内或纯域外;半域内半域外的混合题(如「X 是什么,顺便告诉我明天天气」)三类 difficulty 标不了,出题时直接排除;
- **out_of_domain 剔出四指标门禁与 hit-rate/MRR**:拒答句无法归因到任何检索内容,ContextualRecall 按算法必然趋 0(DeepEval 4.3.4 源码已核实,无豁免路径),混入会互相污染;域外题单独成拒答测试集(10.4);
- 存放:`RAG_GOLDENS_FILE` env 指向 `data/rag_goldens.json`(进 git);test_rag.py 读 env,**医疗/业务切集零代码改动**。

### 10.2 四指标 × 两端映射(引学习计划,已对 DeepEval v4.3.4 源码核实)

```
用户问题 ──→ [检索端] ──→ retrieval_context ──→ [生成端] ──→ 答案
              │                                  │
     ContextualPrecision(查准:捞上来的有用吗)   Faithfulness(忠实:答案有据吗)
     ContextualRecall   (查全:该捞的捞到了吗)   AnswerRelevancy(切题:答的是问的吗)
```

| 指标 | input | actual_output | retrieval_context | expected_output | reference-free? |
|---|:-:|:-:|:-:|:-:|:-:|
| Faithfulness | ✓ | ✓ | ✓ | — | ✅ 可上线 |
| AnswerRelevancy | ✓ | ✓ | — | — | ✅ 可上线 |
| ContextualPrecision | ✓ | — | ✓ | ✓ | ❌ 仅离线 |
| ContextualRecall | ✓ | — | ✓ | ✓ | ❌ 仅离线 |

裁判统一 `bailian_judge("qwen3.7-plus", temperature=0.0)`(`src/ai_evaluation/models.py`,已有;原稿 qwen-plus 在 token-plan 端点实测 404,用户改定 qwen3.7-plus,见附录 E 第三轮)。
易错点照抄学习计划:Faithfulness 也要 input;`retrieval_context` 有序 list[str];Ragas 字段名映射(`user_input`/`response`/`retrieved_contexts`/`reference`)。

### 10.3 非 LLM 检索指标(零裁判成本,检索迭代高频跑)

```
hit-rate@k       = 有任一 expected_doc_id 命中 top-k 的题数 / 门禁题数(剔除 out_of_domain)
per-doc recall@k = mean(top-k 命中的期望文档数 / 期望文档总数)   ← cross_doc 题补充判据
MRR              = mean(1 / 第一个命中 expected_doc_id 的排名)
```

any-hit 对 cross_doc 题乐观(捞到一篇就记 1,查全被高估),per-doc recall@3 补上;分母剔除域外题(它们永远不可能命中,留着会系统性压低指标)。chunk 参数扫描、BM25/rerank 去留这类检索端实验,先跑这些,再跑四指标——省钱且快。

### 10.4 阈值初值与校准

| 项 | 初值 | 校准方式 |
|---|---|---|
| Faithfulness / AnswerRelevancy 门禁 | ≥ 0.8 | 基线分布 + judge reason 抽检 |
| ContextualPrecision / Recall 门禁 | ≥ 0.7 | 同上 |
| 低置信阈值(cosine top-1) | 0.35 | 域外题集上的 cosine 分布;P3 只在最终选定后端校准一套 |
| 域外题拒答率 | ≥ 80% | P3 用拒答集测 |

**拒答率测量 = 裁判二分类(严格语义)**:qwen-plus、temp 0,问「该回答是否明确表示资料不足,**且未给出资料外的实质性内容**?yes/no」——**混合回答(明示拒答 + 夹带编造)判 no**,一个信号同时拉低拒答率、暴露编造;拒答率 = yes 数 / 拒答集题数,单独 assert,reason 留档抽检;字符串匹配只作 sanity 交叉检查,不当判据(模型换措辞就漏判)。Faithfulness 在拒答集上作次要信号,不进拒答门禁。

全部标注「初值」;门禁阈值在基线建立后按实际分布定,不预设结论。cosine 对域外题也可能虚高(嵌入各向异性),校准是必须的不是可选的。

### 10.5 配置对比矩阵(P2 核心产出)

同一 golden set × 四配置(**同题同库**),四指标 + hit-rate@3 + per-doc recall@3 + MRR + 延迟:

| 配置 | CP | CR | F | AR | hit@3 | recall@3 | MRR | p95 延迟 |
|---|---|---|---|---|---|---|---|---|
| keyword-on-语料(业务基线) | | | | | | | | |
| vector | | | | | | | | |
| hybrid(BM25+RRF) | | | | | | | | |
| hybrid + rerank | | | | | | | | |

**基线可比性**:医疗 P0 与语料模式语料不同、题不同,分数不可比——医疗指纹表是学习资产,**不冒充业务基线**;矩阵第一行是 keyword 后端读语料 chunk(bigram 打分)跑同一 golden set 的结果。

**决策规则**:每个组件的保留必须带来可量化增益;无增益 → 砍掉并记录结论。

### 10.6 三模式失败注入(保留现钩子,指纹表引学习计划)

| 模式 | 注入点 | 预期分数指纹 | 诊断认知 |
|---|---|---|---|
| 基线 | 无 | 四指标都高 | — |
| `--bad-retrieval` | 检索器返回无关文档(来源见下) | CP/CR **暴跌**;F **可能依然高** | **忠实 ≠ 正确**:答案可以完全忠实于捞上来的垃圾 |
| `--hallucinate` | 生成 prompt 翻转 | F **跌**;CP/CR **不动** | 检索没病,病在生成端 |

**注入源(跨后端)**:医疗 P0 原样保留固定 3 篇(hypertension/diabetes/medication);语料后端注入 `RAG_BAD_RETRIEVAL_DOC` 指定文档的前 k 个 chunk(缺省 = 语料排序末篇)——固定、无视 question、确定性可复现,与医疗版「固定子集」语义对齐。不用随机抽(可能碰巧相关,污染指纹)、不用按 question 动态挑(钩子就不再是纯注入)。三后端共享同一注入文本。

`--bad-retrieval` 下 Faithfulness 的两种形态都是好结果:LLM 照错误文档答 → F 高 + context 低 = 检索端全责;LLM 说「资料里没有」→ F 也高 = 生成端尽责。**病灶都在检索端**。
升级后三后端各跑一遍三模式,验证指纹不随后端改变——钩子在门面层,注入点与后端无关。

### 10.7 在线评估(reference-free)

- 生产在线没有标准答案,**只能跑 Faithfulness / AnswerRelevancy**(README 4.5 节结论的落地);
- 数据源:门面追加写的 `data/rag_query_log.jsonl`(9.1)——没有日志组件,「对真实问题日志抽样」就是纸面能力;
- 形态:**手动命令** `uv run python -m rag_core.online_sample --n 20`,从日志抽 n 条跑两指标(bailian_judge qwen-plus)+ 统计拒答率、低置信率;不上 cron(个人工具无常驻服务,与 Out 条款「不做 watchdog」精神一致);
- 产出:`data/rag_online_report_<date>.md`(分数表 + 拒答率 + 低置信率 + reason 摘录,gitignore),一条命令出报告;
- Langfuse 生产闭环列为后续演进(学习计划阶段 5),不在本设计内。

---

## 11. 可观测性

### 11.1 span 属性表

| span | 类型 | 记录属性 |
|---|---|---|
| rag_app | agent | trace: input / output / retrieval_context(现版不变) |
| retrieve | retriever | retrieval_context(top-k 文本);后端名;N→(source_path, heading_path) 映射;low_confidence(cosine top-1 阈值);rerank_degraded |
| embed_query | tool | input=question;维度;耗时 |
| vector_search | tool | 候选 id + cosine score(top-20) |
| bm25_search | tool | 分词结果摘要;候选 id + BM25 分 |
| rrf_fuse | tool | 融合序 + RRF 分 |
| rerank | tool | 精排前后序;原始分;是否降级 |
| generate | llm | 现版不变(model/temp 由 observe 自动记录) |
| ingest_corpus(离线) | agent | parse/chunk/embed_batch/upsert 子 tool span;耗时与 token 用量 |

### 11.2 排障工作流

```
答案差 → uv run deepeval inspect 看 trace
  → 四分数指纹定位端(10.6 表):context 低 = 检索端;F 低 = 生成端
  → 检索端:打开 retrieve 子 span,看是哪一路召回丢了目标 doc、
     rerank 是否把对的排掉了、是否 low_confidence
  → 生成端:看 generate span 的 prompt 与输出,对照 claims
```

建索引与查询 trace 分离,互不干扰。

---

## 12. 配置与依赖清单(落地时执行,本次仅清单)

### 12.1 .env 新增项(同步更新 .env.example)

```bash
# ---- RAG 检索后端 ----
RAG_RETRIEVER=keyword            # keyword | vector | hybrid(默认 keyword,基线可复现)
RAG_TOP_K=3
RAG_CORPUS_DIR=./docs            # 语料 = 本仓库 docs/(13 篇 md);真实业务文档到位后换目录
RAG_INDEX_DIR=./data/rag_index
RAG_MANIFEST_FILE=./data/rag_manifest.json
RAG_GOLDENS_FILE=./data/rag_goldens.json   # 医疗 P0 指向 rag_goldens_medical.json,切集零改动
RAG_BAD_RETRIEVAL_DOC=           # 失败注入固定无关文档(缺省 = 语料排序末篇)

# ---- Embedding(百炼按量计费,独立于生成端 token-plan key)----
DASHSCOPE_API_KEY=               # 必填:普通 sk- key(按量计费);token-plan key 实测不可用(附录 E)
EMBEDDING_BACKEND=bailian        # bailian | fake(确定性假向量,零网络零计费,仅 hermetic 测试/无 key 演示)
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
EMBEDDING_BATCH=10               # 官方上限 10 条(v3/v4),单条 8192 token;已实测
EMBEDDING_BASE_URL=              # 缺省 https://dashscope.aliyuncs.com/compatible-mode/v1(双 key 双 endpoint)
EMBEDDING_COST_PER_MILLION_TOKENS=0.5   # 成本估算(ingest 日志用)
EMBEDDING_MAX_RETRIES=5          # 失败重试;429 优先 Retry-After,无则指数退避

# ---- Chunk 切分(字符计量,§4.3 初值,P3 校准)----
RAG_CHUNK_TARGET=700
RAG_CHUNK_MAX=1100
RAG_CHUNK_MIN=140
RAG_CHUNK_OVERLAP=100

# ---- Rerank(DashScope 原生 API,按量计费 key)----
RERANK_MODEL=gte-rerank-v2
RERANK_ENABLED=true
RAG_LOW_CONFIDENCE=0.35          # cosine top-1 阈值(初值,P3 校准);vector/hybrid 共用
RAG_HARD_REFUSE=false            # 硬拒答实验开关,默认关

# ---- 观测 ----
DEEPEVAL_TELEMETRY_OPT_OUT=1     # 关闭 deepeval 匿名遥测
```

### 12.2 依赖增量(uv add)

| 包 | 用途 |
|---|---|
| chromadb | 嵌入式向量库 |
| rank-bm25 | BM25 路 |
| jieba | 中文分词 + 领域词典 |
| markdown-it-py | Markdown 解析 |
| openai / python-dotenv / rich | **转为显式依赖**(现为 deepeval 传递依赖,代码已直接 import,显式化防上游变更断裂) |

dashscope SDK 可选(rerank 也可纯 HTTP 调,冒烟探针已验证纯 HTTP 可行)。按量计费路径下**不引** FlagEmbedding/sentence-transformers;若未来切零计费退路(BGE-M3)再加。

生成端沿用 `OPENAI_BASE_URL`/`OPENAI_API_KEY`(token-plan)——**双 key 双 endpoint**,embedding/rerank 走 `dashscope.aliyuncs.com`。

---

## 13. 分阶段路线与验收

**关键顺序约束:P0 必须先于一切检索升级**——没有基线指纹,「一行不改」与「升级有效」都无从证明。**P0 立刻开工**(医疗语料内嵌,零外部依赖);P1 阻塞于:按量 key 到位(✅ 已完成)+ 30 条 golden 起草且用户重写至少一半 + 语料 docs/ 就位(✅ 已就位)。

| 阶段 | 内容 | 产出 | 验收标准 | 衔接学习计划 |
|---|---|---|---|---|
| **P0 评测先行**(半天–1 天,**立刻开工**) | 在**关键词版**(医疗 6 篇)上写 `test_rag.py`(四指标 + bailian_judge + assert_test),~10 条医疗 golden(agent 起草用户审),跑三模式失败注入 | test_rag.py 全绿;三模式分数对照表 + reason 摘录 | 给一组四分数能盲判病灶端;医疗指纹表 = **学习资产**(不冒充业务基线) | 第 2、3 步本体 |
| **P1 向量检索**(1–2 天) | ingestion + 结构感知切分(字符计量)+ embedding + Chroma;`vector` 后端;keyword 语料模式(bigram);30 条 golden(18/6/6,用户重写至少一半);增量 CLI | rag_core 包;vector 后端;keyword-on-语料 基线;golden set;索引 CLI | **同一份 test_rag.py 零改动**跑 vector,四指标不低于 **keyword-on-语料 基线**;hit-rate@3 相对 keyword 提升;改一个 md 只重嵌该文件;换 embedding 模型名后查询被正确拒绝 | 解耦原则第一次实证 |
| **P2 混合 + 精排**(1 天) | BM25 + jieba(查询/语料共用同一分词函数,初始词典 = 评测术语)+ RRF;gte-rerank 接入(按量 key 已实测调通)与 Identity 降级;`hybrid` 后端 | hybrid 后端;四配置对比矩阵 | CP 相对 dense-only 有可量化提升(**若无提升,依据数据砍掉该组件并记录结论**);失败注入指纹在三后端下均正确;retrieve 端到端延迟记录在案 | 评测驱动选型示范 |
| **P3 生成端与在线化**(半天–1 天) | 引用溯源反查验证;cosine 低置信阈值校准(只在最终选定后端);拒答率裁判二分类测试;在线 reference-free 抽样命令 | 溯源映射;拒答集测试;在线抽样报告 | 答案 `[资料N]` 可反查到文件+小节;域外题拒答率达标(初值 ≥80%,校准);`online_sample --n 20` 一条命令出报告 | 「在线只能 reference-free」落地 |
| **P4 框架对照**(可选,1–2 天) | 同一 golden set 喂 Ragas 同名指标;testset generation 扩充 golden set(标 synthetic);写体检报告 | rag-eval-explained.md(算法差异对照,带源码引用) | 能解释同名指标分差的算法分歧点(含 DeepEval hybrid 模式拆句正则不切中文「。」这类分歧) | 第 4 步本体 |

---

## 14. 风险与开放问题

| 风险 | 影响 | 应对 |
|---|---|---|
| ~~gte-rerank-v2 在 token-plan 套餐的可用性~~ | **已实测不可用**(404/401,附录 E) | **已解决**:按量计费 DASHSCOPE_API_KEY,key 到位后重测通过;Identity 降级兜底 |
| ~~embedding batch 上限待核实~~ | **已实测确认 10 条**(11 条返回 400) | EMBEDDING_BATCH=10 写 config;重试退避;限流参数落地时再测 |
| rerank 分数为请求内相对值(官方) | 固定阈值无依据,强相关实测也仅 ≈0.19 | **已规避**:低置信改 cosine top-1(7.5),rerank 只排序 |
| Token Plan 个人版条款「禁止 API 生产自动化调用」 | 生成端 pytest 循环属灰色地带 | 记录在案,学习/交互式使用自行判断;必要时生成端也迁按量计费 |
| jieba 对领域专名分词质量(语料多英文术语) | BM25 路召回差 | 查询/语料共用同一分词函数;初始词典 = 评测术语;停用词/单字策略 P2 数据定,不行就砍 |
| Chroma 版本升级的持久化兼容 | 索引打不开 | 锁版本;`--rebuild` 分钟级重建,不依赖旧索引可迁移 |
| 裁判成本 | 四指标 × 配置矩阵 × 三模式,调用量不小 | 先跑非 LLM hit-rate/MRR 筛配置,四指标只跑 finalists;temp 0.0 保复现 |
| chunk 参数未经 docs/ 语料校准 | 切分质量不确定 | 全部参数标注初值(字符计量);P1 用 golden set 扫参校准 |
| ~~业务语料尚未入库~~ | 语料已落定 = 本仓库 docs/(13 篇) | 真题域 = AI 评测知识库;若未来有真实业务文档:换 RAG_CORPUS_DIR + 重出题 + --rebuild |
| golden set 出题效度 | 全由 agent 出题 = 我的理解测我的理解 | 用户审核并用自己的措辞重写至少一半(10.1) |

---

## 附录 A:切分前后对照(示例)

原文 `售后政策.md`(节选):

```markdown
# 售后政策
## 退款
### 退款时效
自签收之日起 7 天内可无理由退款,15 天内质量问题换货。
退款到账时效:原路退回 1–3 个工作日。
### 不予退款情形
定制商品、已激活的软件授权不支持无理由退款。
## 发票
### 开票流程
订单完成后在「我的订单」申请,电子发票 24 小时内发送至邮箱。
```

切分结果(目标 ≈700 字符,本例各 section 均低于下限场景按合并规则演示):

| chunk id | heading_path | 文本(头部注入面包屑) |
|---|---|---|
| a1b2c3:0 | 售后政策 > 退款 > 退款时效 | `售后政策 > 退款 > 退款时效\n\n自签收之日起 7 天内……1–3 个工作日。` |
| a1b2c3:1 | 售后政策 > 退款 > 不予退款情形 | `售后政策 > 退款 > 不予退款情形\n\n定制商品……不支持无理由退款。` |
| a1b2c3:2 | 售后政策 > 发票 > 开票流程 | `售后政策 > 发票 > 开票流程\n\n订单完成后……发送至邮箱。` |

注:本例各小节均 <140 字符,实际执行时「退款时效 + 不予退款情形」会按 4.2 规则 3 合并为一个 chunk(同父 H2「退款」);此处拆开仅为展示面包屑注入形态。若某小节含 2000 字长文,则按段落二次切分并加 overlap。

## 附录 B:RRF 计算例

设 k=60,dense 路排名:A=1, B=2, C=3;BM25 路排名:C=1, A=2, D=3。

```
score(A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
score(C) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
score(B) = 1/(60+2)             = 0.01613
score(D) = 1/(60+3)             = 0.01587
融合序:A > C > B > D
```

两路都靠前的 A/C 胜出;只被一路召回的 B/D 仍有机会进候选——这就是「双路保查全、融合保稳健」。

## 附录 C:golden set 样例(3 条,**纯示意**,非真实题集)

> 本附录为售后业务域示意。真实 golden set 域 = AI 评测知识库(语料 = 本仓库 docs/),构成与出题方式见 10.1;g002「拆发票文件」的决定对真实题集不适用,cross_doc 由跨两个真实文件的题保证(schema 校验:≥2 个 expected_doc_ids)。

```json
[
  {
    "id": "g001",
    "question": "签收后第 10 天发现质量问题,还能退吗?",
    "expected_output": "7 天内可无理由退款;15 天内质量问题可换货。第 10 天已过无理由退款期,但仍在质量问题换货期内,可申请换货。",
    "expected_doc_ids": ["售后政策.md"],
    "difficulty": "single_hop"
  },
  {
    "id": "g002",
    "question": "退款多久到账,发票什么时候能开?",
    "expected_output": "退款原路退回 1–3 个工作日;电子发票在订单完成后申请,24 小时内发送至邮箱。",
    "expected_doc_ids": ["售后政策.md"],
    "difficulty": "cross_doc"
  },
  {
    "id": "g003",
    "question": "你们公司明年上市的股价会是多少?",
    "expected_output": "资料中没有相关信息。",
    "expected_doc_ids": [],
    "difficulty": "out_of_domain"
  }
]
```

## 附录 D:三模式指纹速查卡

| 看到什么 | 结论 |
|---|---|
| CP/CR 低,F 高,答案内容错 | 检索端全责:LLM 忠实于垃圾上下文(忠实 ≠ 正确) |
| CP/CR 低,F 高,答案说「资料中没有」 | 检索端全责,生成端尽责拒答 |
| CP/CR 高,F 低 | 生成端病灶:脱离上下文自由发挥,改 prompt |
| CP/CR 高,F 高,AR 低 | 答非所问:检索对、有据,但没回答用户问的点 |
| 四指标都高 | 基线健康——但说明不了评估体系有用,**价值在失败时才显现** |

## 附录 E:冒烟实测记录(2026-11-19)

**第一轮(token-plan key,设计前提被推翻)**:

| 探针 | 结果 |
|---|---|
| text-embedding-v4 @ token-plan compatible endpoint(batch 11/10) | ❌ 404 `model_not_found`——模型在套餐端点不存在 |
| gte-rerank-v2 @ 公共 dashscope endpoint,token-plan key | ❌ 401 `InvalidApiKey`——sk-sp- key 不走按量计费 |
| gte-rerank-v2 @ token-plan 原生 endpoint | ❌ 404 `Model not exist` |

**第二轮(按量计费 DASHSCOPE_API_KEY,普通 sk- key)**:

| 探针 | 结果 |
|---|---|
| embedding batch=11 | ❌ 400「batch size is invalid, it should not be larger than 10」——**上限 10 实测确认** |
| embedding batch=10,dimensions=1024 | ✅ dim=1024,usage 120 token |
| gte-rerank-v2 @ dashscope 原生 HTTP,3 候选 | ✅ 200,usage 63 token;**强相关文档 top-1 relevance_score ≈ 0.188**(相对分数语义的实测佐证:固定 0.3 阈值会把所有查询误判低置信) |

**第三轮(裁判模型可用性,2026-11-19,P0 门禁落地时)**:

| 探针 | 结果 |
|---|---|
| qwen-plus @ token-plan compatible endpoint | ❌ 404 `Model not exist`——裁判原稿模型在套餐端点不存在 |
| qwen-max / qwen-turbo / qwen-plus-latest / qwen-flash 等 @ token-plan | ❌ 404(逐一探测均不存在) |
| qwen4-max-preview @ token-plan | ✅ 可用(生成端现用) |
| qwen3.7-plus @ token-plan | ✅ 可用——**用户改定为裁判**(与被测生成端不同模型,避免自评) |
| qwen-plus @ dashscope 公共端点,DASHSCOPE_API_KEY | ❌ 403 `Free quota exhausted`——按量账户免费额度耗尽,P1/P2 embedding/rerank 前需充值 |

deepeval 4.3.4 会把裁判调用异常静默吞成 score=None(reason/error 均空),门禁全 FAILED 却无原因——`rag_fingerprint.py` 跑三模式前先做裁判冒烟,失败即终止。

**官方文档事实**(help.aliyun.com,2026-11-19 抓取):

- Token Plan(个人/团队)支持模型列表**不含任何 embedding/rerank 模型**;免费额度/按量计费与 Token Plan 额度相互独立;Token Plan 需 sk-sp- key + 专属域名;个人版条款「禁止 API 生产自动化调用」;
- text-embedding-v4:batch 上限 10 条(v3/v4 一致),单条 8192 token,默认 1024 维;0.5 元/百万 token,新人免费额度 100 万 token(90 天,北京);
- gte-rerank-v2:DashScope 原生 HTTP,**不支持 OpenAI 兼容接口**(qwen3-rerank 支持,备选);最大 500 文档/单条 4000 token/请求 30000 token;relevance_score ∈ [0,1] 为**请求内相对分数,不可跨请求绝对比较**;0.8 元/百万 token,无免费额度(2026-10-29 下架);
- rank_bm25 不做分词/预处理,查询必须与语料同一套处理;高频单字 IDF 为负被 epsilon 兜底而非剔除,停用词策略需自行设计。

来源:[文本向量 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api) · [文本排序 API](https://help.aliyun.com/zh/model-studio/text-rerank-api) · [gte-rerank-v2](https://help.aliyun.com/zh/model-studio/gte-rerank-v2) · [模型价格](https://help.aliyun.com/zh/model-studio/model-pricing) · [新人免费额度](https://help.aliyun.com/zh/model-studio/new-free-quota) · [Token Plan 个人版](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview) · [Base URL 总览](https://help.aliyun.com/zh/model-studio/base-url) · [rank_bm25 README](https://github.com/dorianbrown/rank_bm25)