# ai-evaluation

AI 评测项目，使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境（Python 3.12）。

## 目录结构

```
docs/evaluation-frameworks/   # 评测框架调研文档
src/ai_evaluation/            # 评测代码（src 布局）
```

## 常用命令

```bash
uv sync                  # 创建/同步虚拟环境（.venv）
uv add <package>         # 添加依赖
uv add --dev <package>   # 添加开发依赖
uv run python ...        # 在项目环境中运行
```