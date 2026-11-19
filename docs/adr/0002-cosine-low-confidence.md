# 低置信信号用 cosine top-1,不用 rerank 分数

gte-rerank-v2 的 relevance_score 官方定义为「当前请求内的相对分数,不可跨请求绝对比较」;实测强相关文档 top-1 也仅 ≈0.188——在其上设固定阈值会把所有查询误判低置信。决定:低置信 = dense 路 cosine top-1 相似度低于阈值(初值 0.35,域外题集分布校准),vector/hybrid 共用一套;rerank 只负责排序,原始分记 span 供诊断。放弃 rerank 分更强的区分度,换取跨请求可比性与官方语义支持。

## Consequences

- Chroma 返回 cosine distance,相似度 = 1 − distance,勿直接拿 distance 比阈值。
- 阈值只在最终选定后端上校准一套;keyword 后端无分数,不标记。
- 若未来换 rerank 模型(如 qwen3-rerank),先核实其分数语义再考虑改信号源。
