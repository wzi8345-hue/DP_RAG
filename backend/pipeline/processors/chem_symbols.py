"""化学元素符号 ↔ 中文名, 以及表格表头识别/注释工具。

动机 (检索短板修复):
  耐候钢文献的化学成分/力学性能数据几乎都在表格里, 表头是元素符号 (C/Si/Mn...),
  而用户问句用中文 (碳/硅/锰) 且常带"化学成分/含量/质量分数"等词。
  原始 embedding 文本里这些词面零出现 → dense 和 BM25 双路都召不回。
  解决: 入库前给"成分表"表头做中文注释 + 合成一句含"化学成分/含量/质量分数"的描述,
  让查询词面/语义都能挂上钩 (索引侧修复, 同时利好 dense 与 BM25)。
"""

from __future__ import annotations

import re
from typing import List

# 冶金/耐候钢场景常见元素 (符号 -> 中文); 大小写敏感, 仅对"独立成单元格"的符号注释。
ELEMENT_ZH = {
    "C": "碳", "Si": "硅", "Mn": "锰", "P": "磷", "S": "硫", "Cu": "铜",
    "Ni": "镍", "Cr": "铬", "Mo": "钼", "Nb": "铌", "Ti": "钛", "V": "钒",
    "Al": "铝", "Als": "酸溶铝", "N": "氮", "O": "氧", "H": "氢", "B": "硼",
    "Zr": "锆", "Zn": "锌", "Mg": "镁", "Ca": "钙", "Sn": "锡", "Sb": "锑",
    "As": "砷", "W": "钨", "Co": "钴", "Pb": "铅", "Bi": "铋", "Fe": "铁",
    "Ce": "铈", "La": "镧", "Nd": "钕", "Y": "钇", "Ta": "钽", "RE": "稀土",
}

# 组成表判定阈值: 一行里出现 >= N 个元素符号, 视为化学成分表头。
_COMPOSITION_MIN_ELEMENTS = 3


def split_cells(row_text: str) -> List[str]:
    """html_table_to_text 用 ' | ' 连接单元格, 这里拆回来。"""
    return [c.strip() for c in row_text.split("|")]


def is_composition_row(cells: List[str]) -> bool:
    return sum(1 for c in cells if c in ELEMENT_ZH) >= _COMPOSITION_MIN_ELEMENTS


def gloss_cells(cells: List[str]) -> str:
    """给元素符号单元格加中文: ['C','Si','余量'] -> 'C 碳 | Si 硅 | 余量'。"""
    out = []
    for c in cells:
        zh = ELEMENT_ZH.get(c)
        out.append(f"{c} {zh}" if zh else c)
    return " | ".join(out)


def composition_descriptor(header_cells: List[str]) -> str:
    """为成分表合成一行描述, 注入查询常用词 (化学成分/含量/质量分数) + 元素中文。"""
    elems = [f"{c} {ELEMENT_ZH[c]}" for c in header_cells if c in ELEMENT_ZH]
    return "化学成分 含量 (质量分数%) 表; 元素: " + "、".join(elems)


# ---------------------------------------------------------------------------
# 轻量 LaTeX / 数字清洗 (表格单元格里常见 $\leqslant 0.1$、"0 . 1 6" 这类)
# ---------------------------------------------------------------------------

_LATEX_REPL = {
    r"\leqslant": "≤", r"\leq": "≤", r"\geqslant": "≥", r"\geq": "≥",
    r"\approx": "≈", r"\times": "×", r"\pm": "±", r"\%": "%", r"\small": "",
}
_MATHRM_RE = re.compile(r"\\mathrm\{([^}]*)\}")
# "0 . 1 6" -> "0.16"; 仅吃行内空格/制表符, 不跨行 (避免把上一行末尾 "." 与下一行数字粘连)
_NUM_GAP_RE = re.compile(r"(?<=[\d.])[ \t]+(?=[\d.])")


def clean_latex_numbers(text: str) -> str:
    if not text:
        return text
    s = _MATHRM_RE.sub(r"\1", text)
    for k, v in _LATEX_REPL.items():
        s = s.replace(k, v)
    s = s.replace("$", "")
    s = _NUM_GAP_RE.sub("", s)
    return s
