"""失败注入标志的 pytest 注册(评测侧,门面一字不动)。

门面 rag_app 的钩子读 sys.argv(import 时生效,ADR 0001 冻结)。
deepeval test run 把未知选项原样转发给 pytest,但 pytest 默认拒绝
未注册选项——这里注册 --bad-retrieval / --hallucinate 让其通过,
标志留在 sys.argv 里,门面 import 时钩子自动生效:

    uv run deepeval test run src/deepeval_cases/test_rag.py --bad-retrieval
    uv run deepeval test run src/deepeval_cases/test_rag.py --hallucinate

选项本身不被任何测试读取(钩子在门面层,三后端共享注入点)。
"""


def pytest_addoption(parser):
    parser.addoption(
        "--bad-retrieval",
        action="store_true",
        default=False,
        help="失败注入:检索器强制返回无关文档(门面层钩子,经 sys.argv 生效)",
    )
    parser.addoption(
        "--hallucinate",
        action="store_true",
        default=False,
        help="失败注入:生成 prompt 翻转为自由发挥(门面层钩子,经 sys.argv 生效)",
    )