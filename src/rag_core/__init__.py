"""rag_core —— RAG 引擎(架构 §3.2)。

离线 ingestion 流水线:语料 → Chunk → 向量索引(增量 CLI + 真题保鲜)。
在线查询链路的门面在 deepeval_cases/rag_app.py(三签名冻结,ADR 0001)。
"""
