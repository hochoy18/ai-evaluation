# 评估体系与被测系统解耦:门面三签名冻结 + env 后端分发

RAG 检索端要持续升级(keyword → vector → hybrid),而评测代码必须一行不改,「升级有效」才可证明。决定:`rag_app` 门面三签名(retrieve / generate / rag_app)与 `@observe` 装饰器位置冻结,检索上下文保持有序列表;后端经 `RAG_RETRIEVER` env 在门面内分发,评测侧只 import 门面。放弃 import 注入、参数化后端等更灵活的方式,换取「同一份 test_rag.py 零 diff 跑三后端」的可证明解耦。

## Consequences

- 检索内部重构自由,但任何改动不得触及三签名与检索上下文的有序性(排名即数据,ContextualPrecision 按序加权)。
- 失败注入钩子固定在门面层、先于后端分发,三后端共享同一注入点,指纹不随后端改变。
- 「升级有效」由同一真题集上的配置对比矩阵证明;对照基线 = keyword-on-语料,医疗学习基线不冒充业务基线(两者语料与题不同,分数不可比)。
