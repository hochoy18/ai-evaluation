"""Markdown 解析 + 结构感知切分(架构 §4.2)。

markdown-it-py token 流 → H1–H3 section 边界(H4+ 不切)→ 面包屑注入 → Chunk。
以字符计量(len(text),不引 tokenizer);section 文本取源文件行区间,原文原样。
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from rag_core.config import Config


@dataclass
class Chunk:
    """内部协议(架构 §9.2):text 含面包屑头;score 检索端用,建索引时恒 0。"""

    text: str
    metadata: dict
    score: float = 0.0


def doc_id_of(source_path: str) -> str:
    """doc_id = 相对路径稳定 hash(架构 §4.4):增量删改的键,不随内容变。"""
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]


@dataclass
class _Section:
    path: list[str]  # 标题栈 [H1, H2, H3]
    heading_idx: int | None  # 本节标题行行号(合并时作为伪块回注正文)
    blocks: list[tuple[int, int, bool]]  # (起始行, 结束行, 是否原子块)


def _parse_sections(tokens: list[Token], lines: list[str]) -> list[_Section]:
    """H1–H3 开新 section;H4+ 与其余块归入当前 section;空 section 丢弃。"""
    sections: list[_Section] = []
    stack: list[str] = []
    cur: _Section | None = None
    for i, tok in enumerate(tokens):
        if tok.level != 0:
            continue
        if tok.type == "heading_open" and tok.tag in ("h1", "h2", "h3"):
            if cur is not None and cur.blocks:
                sections.append(cur)
            level = int(tok.tag[1])
            stack = stack[: level - 1] + [tokens[i + 1].content]
            assert tok.map is not None
            cur = _Section(path=list(stack), heading_idx=tok.map[0], blocks=[])
            continue
        if tok.type.endswith("_close") or tok.map is None:
            continue
        if cur is None:
            cur = _Section(path=[], heading_idx=None, blocks=[])
        atomic = tok.type in ("fence", "code_block", "table_open", "html_block")
        cur.blocks.append((tok.map[0], tok.map[1], atomic))
    if cur is not None and cur.blocks:
        sections.append(cur)
    return sections


def _split_front_matter(source: str) -> tuple[dict[str, object], str]:
    """首行 ``---`` 起、至闭合 ``---``/``...`` 为 front-matter;只取 title/tags。

    最小 YAML 子集(title 标量、tags 行内列表/逗号串/块列表),不引 PyYAML;
    其余键忽略。front-matter 不进 chunk 正文。
    """
    lines = source.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, source
    end = next(
        (i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None
    )
    if end is None:
        return {}, source
    fm: dict[str, object] = {}
    key: str | None = None
    items: list[str] = []
    for line in lines[1:end]:
        stripped = line.strip()
        if stripped.startswith("- ") and key:
            items.append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        if key and items:
            fm[key] = items
        name, _, rest = stripped.partition(":")
        key, items = name.strip(), []
        rest = rest.strip()
        if not rest:
            continue  # 块列表,后续 "- item" 行收集
        if rest.startswith("[") and rest.endswith("]"):
            rest = rest[1:-1]
        if key == "tags":
            fm[key] = [x.strip().strip("'\"") for x in rest.split(",") if x.strip()]
            key = None
        else:
            fm[key] = rest.strip("'\"")
            key = None
    if key and items:
        fm[key] = items
    return fm, "\n".join(lines[end + 1 :])


def _first_h1(tokens: list[Token]) -> str | None:
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag == "h1":
            return tokens[i + 1].content
    return None


def _render_blocks(blocks: list[tuple[int, int, bool]], lines: list[str]) -> str:
    """块序列 → 源文件行区间原文(首尾空行剥掉,中间空行原样)。"""
    return "\n".join(lines[blocks[0][0] : blocks[-1][1]]).strip("\n")


def _section_body(sec: _Section, lines: list[str]) -> str:
    return _render_blocks(sec.blocks, lines)


def _same_parent(a: _Section, b: _Section) -> bool:
    # 同父小节:标题栈去掉末节后相同;H1 级(父为空)与前言不参与合并
    return len(a.path) >= 2 and a.path[:-1] == b.path[:-1]


def _merge_short_sections(
    sections: list[_Section], lines: list[str], config: Config
) -> list[list[_Section]]:
    """过短 section(< 下限)与相邻同父小节合并,防碎片(架构 §4.2 规则 3)。"""

    def is_short(sec: _Section) -> bool:
        return len(_section_body(sec, lines)) < config.chunk_min

    units: list[list[_Section]] = []
    for sec in sections:
        if is_short(sec) and units and _same_parent(units[-1][-1], sec):
            units[-1].append(sec)
        else:
            units.append([sec])
    # 无前邻同父的短 section,向后并入下一个同父单元
    merged: list[list[_Section]] = []
    for i, unit in enumerate(units):
        if (
            len(unit) == 1
            and is_short(unit[0])
            and i + 1 < len(units)
            and _same_parent(unit[0], units[i + 1][0])
        ):
            units[i + 1] = unit + units[i + 1]
        else:
            merged.append(unit)
    return merged


def _split_blocks(
    blocks: list[tuple[int, int, bool]], lines: list[str], config: Config
) -> list[str]:
    """超上限的块序列按段落(块)边界二次切分:目标 ≈target,相邻 overlap ≈overlap。

    overlap 取上一块末尾的整块(≤overlap 字符),原子块不被腰斩;放不下则放弃。
    """

    def overlap_seed(flushed: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
        seed: list[tuple[int, int, bool]] = []
        for b in reversed(flushed):
            cand = [b] + seed
            if len(_render_blocks(cand, lines)) <= config.chunk_overlap:
                seed = cand
            else:
                break
        return seed

    out: list[str] = []
    cur: list[tuple[int, int, bool]] = []
    seed_only = False  # cur 仅含 overlap 种子时不落块,防纯 overlap 碎片

    def flush() -> list[tuple[int, int, bool]]:
        nonlocal cur, seed_only
        flushed, cur, seed_only = cur, [], False
        if flushed:
            out.append(_render_blocks(flushed, lines))
        return flushed

    for block in blocks:
        if cur and len(_render_blocks(cur + [block], lines)) > config.chunk_max:
            cur = overlap_seed(flush())
            seed_only = True
            if cur and len(_render_blocks(cur + [block], lines)) > config.chunk_max:
                cur, seed_only = [], False  # overlap 放不下则放弃
        cur.append(block)
        seed_only = False
        if len(_render_blocks(cur, lines)) >= config.chunk_target:
            cur = overlap_seed(flush())
            seed_only = True
    if cur and not seed_only:
        flush()
    return out


def _section_pieces(sec: _Section, lines: list[str], config: Config) -> list[tuple[str, bool]]:
    """section → (文本, oversize) 片段:超限原子块独立成片段,其余按段落边界切分。"""
    pieces: list[tuple[str, bool]] = []
    run: list[tuple[int, int, bool]] = []

    def flush() -> None:
        if run:
            pieces.extend((text, False) for text in _split_blocks(run, lines, config))
            run.clear()

    for start, end, atomic in sec.blocks:
        block_text = "\n".join(lines[start:end])
        if atomic and len(block_text) > config.chunk_max:
            flush()
            pieces.append((block_text.strip("\n"), True))
        else:
            run.append((start, end, atomic))
    flush()
    return pieces


def chunk_document(source: str, source_path: str, config: Config) -> list[Chunk]:
    """一个源文档 → Chunk 列表(面包屑头 + §4.4 metadata schema)。"""
    fm, body = _split_front_matter(source)
    lines = body.splitlines()
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(body)
    fm_title = fm.get("title")
    title = (
        fm_title
        if isinstance(fm_title, str) and fm_title
        else (_first_h1(tokens) or Path(source_path).stem)
    )
    fm_tags = fm.get("tags")
    tags = ",".join(str(t) for t in fm_tags) if isinstance(fm_tags, list) else ""
    doc_id = doc_id_of(source_path)

    chunks: list[Chunk] = []

    def emit(breadcrumb: str, body: str, oversize: bool) -> None:
        # 无法按块边界再切的超限正文(如单个超长段落)同样标 oversize,不留未标记的超限 chunk
        oversize = oversize or len(body) > config.chunk_max
        text = f"{breadcrumb}\n\n{body}"
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "title": title,
                    "heading_path": breadcrumb,
                    "chunk_index": len(chunks),
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    "tags": tags,
                    "oversize": oversize,
                },
            )
        )

    for unit in _merge_short_sections(_parse_sections(tokens, lines), lines, config):
        if len(unit) == 1:
            sec = unit[0]
            breadcrumb = " > ".join(sec.path) or title
            for body, oversize in _section_pieces(sec, lines, config):
                emit(breadcrumb, body, oversize)
        else:
            # 合并单元:面包屑 = 共同父路径;各节标题行作为伪块回注正文,
            # 整体走同一切分器——超限原子块仍独立、超上限仍二次切分,合并不绕过规则
            breadcrumb = " > ".join(unit[0].path[:-1]) or title
            blocks: list[tuple[int, int, bool]] = []
            for sec in unit:
                if sec.heading_idx is not None:
                    blocks.append((sec.heading_idx, sec.heading_idx + 1, False))
                blocks.extend(sec.blocks)
            merged = _Section(path=unit[0].path[:-1], heading_idx=None, blocks=blocks)
            for body, oversize in _section_pieces(merged, lines, config):
                emit(breadcrumb, body, oversize)
    return chunks
