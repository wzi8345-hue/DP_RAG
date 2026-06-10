"""Function Calling 工具 schema (v4).

设计原则:
1. 字段名极短 (t/kw/docs/refs/figs/...), 节省 LLM 输出 token;
2. description 详细完整, 严格对齐 prompts/router_rules.md 的语义契约;
3. 单轮 FC 即出全部决策, 不做 ReAct 多轮;
4. 路径类型用 oneOf + 字面常量 t 字段区分, schema 层面就消除"metadata 不该带 rewrites"
   "metadata 必须有 filter" 等硬约束 (无需 _validate_decision 防御);
5. router 3 个工具 (plan/multi/ask) 和 reflect 3 个工具 (ok/retry/partial), 各自
   tool_choice="required", LLM 必须且仅能选一个;
6. 不引入 HyDE: 文献检索领域 LLM 假设回答会自信地把幻想拉进召回, 风险高于收益。

与现有 RouteDecision (models.py) 的字段映射:
  paths[].t == "summary"     →  routes: ["summary"]
  paths[].t == "progressive" →  routes: ["progressive"]
  paths[].t == "local"       →  routes: ["local"]
  paths[].t == "metadata"    →  routes: ["metadata"]

  paths[].kw       → rewrites[route]   (列表内部以空格 join 成单字符串)
  paths[].docs     → target_docs
  paths[].refs     → doc_refs → 在 decision_builder 内用 doc_registry 查表追加 target_docs
  paths[].figs     → fig_refs
  paths[].tabs     → table_refs
  paths[].pages    → page_refs
  paths[].paras    → paragraph_refs
  paths[].ents     → entities
  paths[].ctype    → chunk_type   (progressive / local / metadata 均可带, enum:
                                    image | table | equation | references)
  paths[].expand   → expand_neighbors (各 path 的 expand 合并去重, enum:
                                    assets | adjacent | page | similar)

  顶层 time        → time
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .limits import DEFAULT_ROUTING_LIMITS, RoutingLimits


# ---------------------------------------------------------------------------
# 邻域扩展 (依赖图谱场景): 检索完成后沿 chunk 间的边各扩 1 跳
# ---------------------------------------------------------------------------

_EXPAND_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "string",
        "enum": ["assets", "adjacent", "page", "similar"],
    },
    "description": (
        "【可选, 依赖图谱式邻域扩展】仅当用户意图是'从当前内容向外扩散'时填; 否则**省略**。"
        "检索到种子 chunk 后, 系统会沿对应的边各扩 1 跳并回填:\n"
        "  - assets  : 沿图/表/公式 ↔ 正文的交叉引用边 (如问图表时连带其讲解段落)\n"
        "  - adjacent: 同篇相邻段落 (上下文邻域); 用户说'还研究了什么/相关的内容/前后文'\n"
        "  - page    : 同篇同页其它内容; 用户说'图N附近的文字/这页还有什么'\n"
        "  - similar : 跨文献的同类/相似内容 (more-like-this); 用户说'其他类似方法/类似的研究'\n"
        "典型组合: '图3附近的文字'→['page','assets']; '还研究了什么'→['adjacent','assets']; "
        "'其他类似方法'→['similar']。普通事实/机制类查询不要填 expand。"
    ),
}


_KW_FIELD_DESC = (
    "离散关键词数组 (不是整句). 结构=**主体+问点**: 材料牌号/元素/工艺/软件 + 含量/规格/机理/显著性等; "
    "剥掉疑问词/口语. **禁止** 文献/论文/研究/资料/有没有/哪些/关于/方面等元话语 "
    "(路径已由 t 表达, 写进 kw 会在大库 BM25 命中海量摘要块). "
    "**硬约束**: 用户发话含汉字时, kw 必须同时含中文词与对应英文术语/缩写 (中英成对, 语料多为英文). "
    "summary: 领域主题词+英文同义词; progressive: 主体+属性中英成对; local: 问点属性词+英文同义词."
)


# ---------------------------------------------------------------------------
# 子 schema: 4 条路径 (oneOf items)
# ---------------------------------------------------------------------------

_PATH_SUMMARY: Dict[str, Any] = {
    "type": "object",
    "description": (
        "summary 路径: 在文献摘要 (abstract/summary) 级别就能回答的查询, 不需要 chunk 级细节。"
        "三类典型场景:\n"
        "  (1) 总结/汇总/对比/概述 (例: '总结一下这几篇论文'/'对比 X 与 Y 的方法');\n"
        "  (2) 【文献发现/筛选/盘存】用户在问'有没有/哪些/是否存在' 关于某主题的文献, "
        "或'X 相关的资料/论文/研究有什么' (例: '有没有腐蚀钢相关的文献资料'/'哪些论文涉及 LiFePO4 的循环寿命'/'有什么关于钒电池的研究');\n"
        "  (3) 探索性查询: 用户对某领域还不熟, 想先看面再钻细节。\n"
        "命中以上任一场景: 通常只用 summary 一条路径; 若用户后续追问具体细节, 下一轮再补 progressive/local。"
        "判别要点: 用户问的是'文献是否存在/有什么'而不是'某事实是什么/某机制怎样' → summary。\n"
        "回指支持: 若用户基于上一轮文献列表说'总结一下第1、3篇'/'对比上面那两篇', "
        "应填 refs=[1,3] 而不是把整段文献名抄进 docs (与 local 路径一致)。"
    ),
    "properties": {
        "t": {"const": "summary"},
        "kw": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": _KW_FIELD_DESC,
        },
        "docs": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "可选: 限定 summary 在某些已知文献内 (整篇标题串)。与 refs 二选一; "
                "上一轮列表里有的优先 refs 编号回指。"
            ),
        },
        "refs": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": (
                "可选: 上一轮文献列表的 1-based 编号数组。用户说'总结第1篇'/'对比上面 1 和 3' "
                "时填; 系统自动把编号解析为 doc_name 追加到 docs。越界编号会被丢弃。"
            ),
        },
    },
    "required": ["t", "kw"],
    "additionalProperties": False,
}


_PATH_PROGRESSIVE: Dict[str, Any] = {
    "type": "object",
    "description": (
        "progressive 路径: 用户问【具体的子问题/事实/机制/数据/结论】, 但没指定特定文献。"
        "两级检索: 先按 kw 找候选文献, 再在候选内做 chunk 级精排。\n"
        "典型场景:\n"
        "  - '腐蚀钢在海洋环境下的腐蚀机理是什么';\n"
        "  - '钒电池的最高循环寿命达到多少';\n"
        "  - '什么因素影响 LiFePO4 的高温循环性能';\n"
        "  - '有没有引用 XX 的参考文献' (此时配合 ctype=references)。\n"
        "判别要点: query 中有'是什么/为什么/怎么样/多少/如何' 等指向具体事实/机制/数据的疑问词 → progressive。\n"
        "【严禁】用于纯文献发现查询 ('有没有/哪些 X 文献' 是 summary 的场景, 不是 progressive)。"
    ),
    "properties": {
        "t": {"const": "progressive"},
        "kw": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": _KW_FIELD_DESC,
        },
        "ctype": {
            "type": "string",
            "enum": ["image", "table", "equation", "references"],
            "description": (
                "可选: chunk 类型过滤. 通常 progressive 不带 ctype (从正文池召回); "
                "当用户问公式/图/表时填对应值. "
                "ctype='references' **仅当**用户明确索取'引用/参考了哪些文献'(即要引文列表本身) "
                "时填, 如 '有没有引用了 XX 的参考文献'/'看下 references/bibliography'. "
                "【严禁误填】只提到某标准/方法/试验/规范名 (如 'ASTM 标准方法'/'用什么方法表征 X') "
                "不是问参考文献, 是正文事实问句 → 省略 ctype. "
                "ctype=references 时系统会在 references chunk 池里召回 (不需要给条目编号)."
            ),
        },
        "expand": _EXPAND_SCHEMA,
    },
    "required": ["t", "kw"],
    "additionalProperties": False,
}


_PATH_LOCAL: Dict[str, Any] = {
    "type": "object",
    "description": (
        "local 路径: 用户已指定文献 (整篇标题→docs), 或回指上一轮列表里的某篇 (用 refs 编号)。"
        "在指定文献内做 chunk 级检索, 不做候选文献筛选。"
    ),
    "properties": {
        "t": {"const": "local"},
        "kw": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                _KW_FIELD_DESC + " 文献已锁定: **不要**重复文献全名, 只写问点属性词."
            ),
        },
        "docs": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "目标文献的完整标题串数组。与 refs 二选一; 若上一轮列表里就有用户回指的那篇, "
                "优先用 refs 编号回指, 不要重复写整段文献名。"
            ),
        },
        "refs": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": (
                "上一轮检索结果文献列表的 1-based 编号数组。仅当上一轮列表非空且用户用 "
                "'第1篇'/'上面那篇'/'刚才的2和3'/'它/这篇' 等回指措辞时填写。系统会自动把编号"
                "解析为对应的 doc_name 追加到 docs 里。越界编号会被丢弃。"
            ),
        },
        "ctype": {
            "type": "string",
            "enum": ["image", "table", "equation", "references"],
            "description": (
                "可选: chunk 类型过滤. 用户在某篇文献里专门问公式/图/表时填对应值. "
                "ctype='references' **仅当**用户明确索取该文献的引文列表本身时填, "
                "如 '这篇文献引用了哪些参考文献'/'它的 references 有哪些' (会全量召回该文献"
                "所有 references chunk, 不需要给条目编号). "
                "【严禁误填】在某篇里问标准/方法/试验/数值 (如 '这篇用哪个 ASTM 标准方法') "
                "是正文事实问句 → 省略 ctype, 不要当成参考文献."
            ),
        },
        "expand": _EXPAND_SCHEMA,
    },
    "required": ["t", "kw"],
    "additionalProperties": False,
}


_PATH_METADATA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "metadata 路径: 仅用于硬过滤命中。用户明确说 '图N'/'表N'/'第N页'/'第N段', 或要求'正文"
        "精确查找某术语'时使用。注意: metadata 不接受关键词改写 (schema 里也没 kw 字段), "
        "完全靠 figs/tabs/pages/paras/ents 这些硬过滤字段表达检索意图; 必须至少给一个 filter, "
        "否则不要选此路径。chunk_type 可选: image (问图)/table (问表)。"
    ),
    "properties": {
        "t": {"const": "metadata"},
        "figs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "图编号数组 (字符串), 如 ['1','2','3a']。",
        },
        "tabs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "表编号数组 (字符串), 如 ['1','2']。",
        },
        "pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "用户明确提到的页码 1-based 数组。",
        },
        "paras": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "用户明确提到的段落编号 1-based 数组。",
        },
        "ents": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "正文精确子串匹配的术语/实体名数组。一般来自用户引号引起的字符串, "
                "如 '查找 \"LiNiCoMnO2\"' → ents=['LiNiCoMnO2']。"
            ),
        },
        "ctype": {
            "type": "string",
            "enum": ["image", "table", "equation", "references"],
            "description": (
                "chunk 类型过滤; 问图填 image, 问表填 table, 问公式填 equation. "
                "references 一般通过 progressive/local 路径配合 ctype 处理, metadata "
                "路径里很少用 references (除非要在某篇文献的 references chunk 上做"
                "其他硬过滤, 如 ents)."
            ),
        },
        "docs": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "可选: 限定 metadata 命中在某些文献内 (如 'X 文献第3页')。"
            ),
        },
        "refs": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": (
                "可选: 上一轮列表的 1-based 编号, 等价于 docs (系统会自动转 doc_name)。"
            ),
        },
        "expand": _EXPAND_SCHEMA,
    },
    # 硬约束: 必须至少给一个 filter, 否则不允许选 metadata
    "anyOf": [
        {"required": ["t", "figs"]},
        {"required": ["t", "tabs"]},
        {"required": ["t", "pages"]},
        {"required": ["t", "paras"]},
        {"required": ["t", "ents"]},
    ],
    "additionalProperties": False,
}


_PATH_ONEOF: Dict[str, Any] = {
    "oneOf": [_PATH_SUMMARY, _PATH_PROGRESSIVE, _PATH_LOCAL, _PATH_METADATA],
    "description": "单条检索路径; 通过 t 字段区分类型。",
}


def _retrieve_bias_schema() -> Dict[str, Any]:
    return {
        "type": "string",
        "enum": ["semantic", "entity_heavy", "keyword", "balanced"],
        "description": (
            "可选。本 query 的 hybrid 检索偏好 (仅影响 progressive/local 的 dense+bm25 融合权重; "
            "summary/metadata 忽略):\n"
            "- semantic: 机理/影响/对比/总结等抽象概念问句\n"
            "- entity_heavy: 化学式/专名/DOI/引号术语/参考文献(recall references chunk)\n"
            "- keyword: 短 query/文献名片段/图表页码/标题词\n"
            "- balanced: 不确定或不填 (系统按 balanced 映射)"
        ),
    }


def _rerank_mode_schema() -> Dict[str, Any]:
    return {
        "type": "boolean",
        "enum": [True],
        "description": (
            "【可选, 非 required】仅当 rerank 精排应使用 paths 的 kw rewrite、而非用户原话时输出 true。\n"
            "默认行为: **省略本字段** → rerank 用用户原始发话 (多数场景)。\n"
            "何时输出 true (满足任一即可):\n"
            "  - 用户发话极泛/无检索锚点: '有没有/哪些/介绍一下/这方面/怎么样/有哪些研究' 等, "
            "具体实体已写进 kw\n"
            "  - 原话仍含未消解代词 ('这篇/上面/它'), 但 kw/docs/refs 已替换为明确实体\n"
            "  - kw 相对原话做了显著扩写 (同义词/英文/化学式/缩写), 原话不足以做 chunk 相关性判断\n"
            "何时 **不要** 输出 (直接省略, 禁止输出 false):\n"
            "  - 用户问句已含明确主题+问点, 可直接精排: '钒电池最高循环寿命是多少'\n"
            "  - metadata 路径且 filters 已锁定图/页/实体 (精排靠 filter+原话即可)\n"
            "  - reuse 模式 (不检索, 无 rerank)\n"
            "注意: kw 仍只用于召回; rerank_mode=true 时精排才改用 kw 拼接串, 不是额外字段。"
        ),
    }


def _paths_schema(*, max_paths: int) -> Dict[str, Any]:
    cap = max(1, int(max_paths))
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": cap,
        "items": _PATH_ONEOF,
        "description": (
            f"1-{cap} 条检索路径; 通常 1 条够用。"
            "同一意图可组合 progressive/local + metadata (互补双路径); "
            "多个互斥 filter (不同图/页/参考文献) 必须拆到 multi 工具的各 sub, 不要硬塞进 plan。"
        ),
    }


def _sub_strategy_schema(*, max_paths: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "description": "一个子查询的检索策略, 字段语义与 plan 一致 (除 conf 外)。",
        "properties": {
            "paths": _paths_schema(max_paths=max_paths),
            "time": {"type": "string"},
            "retrieve_bias": _retrieve_bias_schema(),
            "rerank_mode": _rerank_mode_schema(),
            "id": {
                "type": "string",
                "description": "可选子查询 id; 不填则系统按顺序 sub1/sub2/sub3。",
            },
        },
        "required": ["paths"],
        "additionalProperties": False,
    }


def build_plan_tool(*, limits: Optional[RoutingLimits] = None) -> Dict[str, Any]:
    lim = limits or DEFAULT_ROUTING_LIMITS
    max_p = lim.max_paths_per_sub
    return {
        "type": "function",
        "function": {
            "name": "plan",
            "description": (
                "【90% 场景调这个】为单一意图的查询提交一个完整检索策略, 单次 LLM 调用内一次性产出。"
                "\n"
                "适用场景:\n"
                "  - 单一主题/单一文献/单一图表的查询\n"
                "  - 即使含代词或回指, 也可在 kw/docs/refs 字段里**直接**填消解后的实体\n"
                "  - 即使是泛 query (如 '讲讲这些论文'), 只要锁定到了具体文献或主题, 也走 plan\n"
                "\n"
                "选路规则 (与 router_rules.md 完全一致):\n"
                "  - summary    : 总结/汇总/对比/概述; 摘要可答→只用此路径\n"
                "  - progressive: 问细节, 未指定文献\n"
                "  - local      : 已知具体文献 (docs) 或回指上一轮列表 (refs)\n"
                "  - metadata   : 明确说图N/表N/第N页/第N段, 或要求精确实体查找; 必须有 filter, 不接受 kw\n"
                "\n"
                "改写规则:\n"
                "  1. kw=主体+问点 离散数组; 禁止 文献/研究/有没有/哪些 等元话语 (见 router_rules)\n"
                "  2. 代词必须在 kw/docs/refs 里消解为明确实体或编号\n"
                f"  3. 路径最多 {max_p} 条 (互补双路径); 互斥 filter 必须拆 multi\n"
                "  4. rerank_mode: 问句已含牌号+具体问点时**省略**; 极泛发话才 true\n"
                "\n"
                "复合查询 (含 2+ 个独立意图 / 互斥 filter, 如 '图3说明 + 第5页表 + 参考文献X') "
                "必须走 multi 工具, 每个意图一个 sub, 禁止硬塞进 plan。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": _paths_schema(max_paths=max_p),
                    "time": {
                        "type": "string",
                        "description": (
                            "时间过滤表达式; 仅当 query 显式含年份/时间窗口时填。"
                            "格式: 'YYYY-YYYY' (区间, 如 '2020-2026') 或 'YYYY' (单年)。"
                            "用户说 '近三年'/'近5年' 时自行计算 (例: 当前 2026 年, '近三年'→'2023-2026')。"
                        ),
                    },
                    "retrieve_bias": _retrieve_bias_schema(),
                    "rerank_mode": _rerank_mode_schema(),
                },
                "required": ["paths"],
                "additionalProperties": False,
            },
        },
    }


def build_multi_tool(*, limits: Optional[RoutingLimits] = None) -> Dict[str, Any]:
    lim = limits or DEFAULT_ROUTING_LIMITS
    max_p = lim.max_paths_per_sub
    max_subs = lim.max_subqueries
    return {
        "type": "function",
        "function": {
            "name": "multi",
            "description": (
                "【复合查询专用】当 query 含 2+ 个独立检索意图时使用。识别标志:\n"
                "  - 用 '和/与/再/同时/分别/对比/以及' 连接 2+ 不同主题\n"
                "  - 明确提到 2 篇以上文献, 且每篇要查不同东西\n"
                "  - 同时问 (图/表/页) 又问 (正文细节/参考文献) — 每个意图一个 sub\n"
                "\n"
                "示例:\n"
                "  - 'X 文献里图 3 的内容, 再讲讲 Y 文献的方法' → 拆 2 子查询\n"
                "  - '图3的说明、第5页的数据表、以及参考文献中关于X的引用' → 拆 3 子查询:\n"
                "      sub1: metadata+figs=[3]; sub2: metadata+pages=[5]+ctype=table; "
                "sub3: progressive/local+ctype=references+kw=[X]\n"
                "  - '总结这三篇, 并列出共同实验条件' → 拆 2 子查询 (summary + metadata/ents)\n"
                "\n"
                "若 query 只有一个意图 (即使带多个关键词或复杂修饰), 用 plan 而非 multi。"
                f"子查询数量上限 {max_subs} (超过说明 query 太复杂, 考虑反问 user)。"
                f"每个 sub 内 paths 上限 {max_p}。"
                "\n"
                "synth 字段给上下文构建器一个拼接提示, 决定输出顺序/分栏方式; 不写则默认按 subs 顺序拼接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subs": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": max_subs,
                        "items": _sub_strategy_schema(max_paths=max_p),
                        "description": (
                            f"子查询数组 (2-{max_subs}), 每个子查询独立检索且 filters 互不合并, "
                            "结果按 id (或顺序) 分组。"
                        ),
                    },
                    "synth": {
                        "type": "string",
                        "description": (
                            "可选: 告诉 context_builder 如何拼接各 sub 的结果 "
                            "(例: '先列图3说明, 再给第5页表, 最后列参考文献')."
                        ),
                    },
                },
                "required": ["subs"],
                "additionalProperties": False,
            },
        },
    }


# 默认工具 (向后兼容静态 import)
PLAN_TOOL: Dict[str, Any] = build_plan_tool()
MULTI_TOOL: Dict[str, Any] = build_multi_tool()


# ---------------------------------------------------------------------------
# Router Tool 3: ask (反问 user; 仅极端情况)
# ---------------------------------------------------------------------------

ASK_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask",
        "description": (
            "【兜底, 默认 enable_ask=false 时不暴露给 LLM】仅在以下全部满足时调用:\n"
            "  1. query 极度模糊 (例: '讲讲'/'怎么样'/'呢') 且信息密度 < 3 个名词\n"
            "  2. history 中没有可锚定的主题/文献\n"
            "  3. doc_registry 为空 (没有上一轮文献可回指)\n"
            "  4. 强行选 progressive 也无法生成有意义的 kw\n"
            "\n"
            "调用后系统会跳过本轮检索, 把 q 反问给 user。\n"
            "若仅是 '泛 query 但有 history 锚定'（如 history 里提了 X 文献), 不要 ask, 用 plan + "
            "summary/local 路径处理。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "反问给 user 的问题, 应明确指出缺失的信息维度。",
                },
                "opts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选: 给 user 的 2-4 个候选选项, 帮助他快速确认意图。",
                },
            },
            "required": ["q"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Router Tool 4: reuse (不检索, 直接复用上轮 / 直接回答 / 礼貌拒答)
# ---------------------------------------------------------------------------

REUSE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reuse",
        "description": (
            "【不进行新检索, 直接生成最终答复】仅当用户发话满足以下全部条件时使用:\n"
            "  1. 用户明确指代上轮内容 (使用'根据上面内容/根据上面的信息/刚才说的/上面那个/之前提到的'等指代表达), "
            "或者发话属于闲聊/超界/确认/继续等非检索意图;\n"
            "  2. 用户发话不包含任何新的检索要素 (新关键词/新文献名/新图表号等);\n"
            "  3. 所需信息可以完全从上下文历史对话的总结结果中获取, 不需要额外的文献检索。\n"
            "\n"
            "适用场景 (按 mode 区分):\n"
            "  - reformat   : 把上一轮已经给出的答案换种表达 — '用通俗的话说' / "
            "'翻译成英文' / '压成 100 字' / '换个口吻再说一遍'.\n"
            "  - drilldown  : 用户**明确指代上轮内容**并要求展开/解释 — '根据上面内容解释一下X' / "
            "'上面说的那个X是什么意思' / '根据刚才的总结展开第3点'. "
            "**必须包含明确的指代词** ('上面/刚才/之前/根据上面内容/根据上面的信息'), "
            "若用户只是问'X是什么'而没有指代上轮, 则不算 drilldown, 应走 plan/multi.\n"
            "  - metasession: 询问会话/检索状态本身 — '刚才给我看了哪几篇' / "
            "'我们聊到哪里了' / '上一轮你检索了什么'.\n"
            "  - confirm    : 用户在确认上轮答案 — '是这样吗?' / '确定吗' / '对吗'.\n"
            "  - continue   : 让上轮答案继续往下说 — '继续' / '接下去呢' / 'go on'.\n"
            "  - chitchat   : 闲聊 / 致意 / 不涉及检索 — '你好' / '谢谢' / "
            "'你能做什么' / '辛苦了'.\n"
            "  - out_of_scope: 显然超出文献知识库范围的请求 — '今天天气' / "
            "'帮我写一段 Python' / '帮我算 2^32' / 'DOI 是什么意思' (元概念).\n"
            "\n"
            "【硬约束 — 违反任一条即禁止选 reuse】\n"
            "  1. 只要用户发话里出现新的检索要素 (新关键词/新文献名/新图表号/'有没有X文献'/'图N'/'表N'/'第N页'), "
            "立即放弃 reuse, 改走 plan/multi。\n"
            "  2. 若用户发话没有明确指代上轮内容 (无'上面/刚才/之前/根据上面/根据上面的'等指代词), "
            "且发话包含疑问词 ('是什么/怎样/为什么/多少/有没有'), 则不是 reuse, 应走 plan/multi。\n"
            "  3. 即使发话包含指代词, 若追问的内容超出上轮 context/answer 的覆盖范围 "
            "(需要新的事实/数据/文献才能回答), 也必须走 plan/multi, 不能用 drilldown。\n"
            "\n"
            "【判别决策流程 (按顺序逐条检查)】\n"
            "  1. 发话是否为闲聊/致意/超界? → 是: reuse (chitchat/out_of_scope)\n"
            "  2. 发话是否为确认/继续? → 是: reuse (confirm/continue)\n"
            "  3. 发话是否包含明确指代词 ('上面/刚才/之前/根据上面/根据上面的/上面说的')? "
            "→ 否: 走 plan/multi (不是 reuse)\n"
            "  4. 发话是否包含新的检索要素? → 是: 禁止 reuse, 走 plan/multi\n"
            "  5. 所需信息是否可从上轮 context/answer 完整获取? → 否: 禁止 reuse, 走 plan/multi\n"
            "  6. 以上均通过 → 可走 reuse (reformat/drilldown/metasession)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [
                        "reformat", "drilldown", "metasession",
                        "confirm", "continue", "chitchat", "out_of_scope",
                    ],
                    "description": "上面 7 种之一; 不允许其他值。",
                },
                "op": {
                    "type": "string",
                    "description": (
                        "≤ 60 字的简短指令, 告诉生成模型对上轮内容要做什么操作。"
                        "示例: 'translate into English' / '压缩到 80 字' / "
                        "'概括该文献主要内容' / '礼貌告知此问题不在知识库覆盖内'。"
                        "禁止重复用户原话, 只写要执行的动作。"
                    ),
                },
                "refs": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": (
                        "【可选, 非 required】doc_registry 1-based 编号, 锁定 reuse 针对的文献 "
                        "(1 篇或多篇); 与 plan.paths[].refs 同一编号体系。\n"
                        "应填 refs 的情况 (doc_registry 非空时):\n"
                        "  - drilldown/reformat/continue: 用户回指具体文献 "
                        "('这篇/那篇/出处/上面那篇/第N篇/它')\n"
                        "  - metasession: 用户问某篇文献本身 ('这篇论文讲了什么'), 不是问会话状态\n"
                        "可省略 refs 的情况:\n"
                        "  - chitchat / out_of_scope / confirm\n"
                        "  - metasession 且问'刚才检索了哪几篇/聊到哪里了' (列全部, 不锁定)\n"
                        "  - reuse 面向整个上轮 answer, 非某一文献\n"
                        "禁止输出 false; 多篇用数组如 [1,3]。"
                    ),
                },
            },
            "required": ["mode", "op"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Reflect Tool 1: ok (结果足够, 不重试)
# ---------------------------------------------------------------------------

OK_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ok",
        "description": (
            "评估结论: 当前检索结果**充分、相关、数量足够**, 可以直接进入上下文构建。\n"
            "\n"
            "判断标准 (与 reflect_system.md 一致):\n"
            "  - 覆盖度: 结果涵盖了问题涉及的关键概念/实体 (对比类两边都召回了)\n"
            "  - 相关度: chunk 内容对准了问题诉求 (问数值确实给了数值, 问图表确实给了图表)\n"
            "  - 充分度: 结果数量 ≥3 条且不全是碎片\n"
            "\n"
            "**倾向规则**: 不确定时优先选 ok, 避免过度重试浪费 LLM 调用次数。\n"
            "判断理由请写在 thinking 块里, 此工具不接受任何参数。"
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Reflect Tool 2: retry (重试, 给出新策略)
# ---------------------------------------------------------------------------

def _retry_plan_shape(*, limits: Optional[RoutingLimits] = None) -> Dict[str, Any]:
    plan = build_plan_tool(limits=limits)
    multi = build_multi_tool(limits=limits)
    return {
        "oneOf": [
            plan["function"]["parameters"],
            multi["function"]["parameters"],
        ],
        "description": (
            "新策略, 与 plan 或 multi 工具的 parameters 完全同构。意识到原 query 是复合的 → 用 multi 形态; "
            "其他情况用 plan 形态。"
        ),
    }


def build_retry_tool(*, limits: Optional[RoutingLimits] = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "retry",
            "description": (
                "评估结论: 当前结果**不足**, 提供完全新的检索策略重试一次。\n"
                "\n"
                "硬约束 (避免无效重试):\n"
                "  1. 新策略必须与上一轮**显著不同** (换路径 / 换 kw / 加/松 filter / 拆 subquery)\n"
                "  2. 上一轮哪条路径召回为 0 或全不相关 → 这一轮换另一条路径或加 filter 收紧\n"
                "  3. 若 query 涉及已知文献 (本轮已检索文献列表给出), 强烈推荐 local + refs\n"
                "  4. 单次反思只能重试一次, 若已是第二轮反思机会, 应该选 partial 而非 retry\n"
                "\n"
                "cause 字段帮助系统分类无效重试 (用于指标):\n"
                "  - zero    : 上一轮 0 命中\n"
                "  - off     : 召回有内容但全部偏题\n"
                "  - narrow  : 召回数量少, 需放宽 (例: 去 filter, 换 progressive)\n"
                "  - broad   : 召回过多噪音, 需收紧 (例: 加 time/ents filter, 换 local)\n"
                "  - compound: 意识到原 query 实际是复合的, 没拆 → 新策略用 multi 形态"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cause": {
                        "type": "string",
                        "enum": ["zero", "off", "narrow", "broad", "compound"],
                        "description": "重试原因分类码。",
                    },
                    "plan": _retry_plan_shape(limits=limits),
                },
                "required": ["cause", "plan"],
                "additionalProperties": False,
            },
        },
    }


# 默认 reflect retry (向后兼容)
_RETRY_PLAN_SHAPE: Dict[str, Any] = _retry_plan_shape()
RETRY_TOOL: Dict[str, Any] = build_retry_tool()


# ---------------------------------------------------------------------------
# Reflect Tool 3: partial (放弃重试, 标记部分回答)
# ---------------------------------------------------------------------------

PARTIAL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "partial",
        "description": (
            "评估结论: 已尽力但仍不理想 (例: 知识库可能不含该信息; 上一轮已重试过; "
            "尝试任何新策略都很可能再次失败)。\n"
            "\n"
            "调用后 context_builder 会附 '信息有限' 提示给生成阶段, 让最终 answer 显式说明"
            "知识库覆盖不足, 避免 LLM 强行编造。\n"
            "\n"
            "仅在以下任一情况选 partial, 不要轻易放弃:\n"
            "  - 已用尽 max_retries 重试预算\n"
            "  - 上一轮已 retry 过且结果仍 0 命中或全不相关\n"
            "  - query 涉及的实体/概念在已检索文献中根本不存在"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "缺口说明 (≤80 字), 会注入到 context 末尾给生成模型参考。",
                }
            },
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# 工具集打包
# ---------------------------------------------------------------------------

def router_tools(
    enable_multi: bool = True,
    enable_ask: bool = False,
    enable_reuse: bool = True,
    *,
    limits: Optional[RoutingLimits] = None,
) -> List[Dict[str, Any]]:
    """返回 router 的工具列表。

    Args:
        enable_multi: 是否暴露 multi (复合查询).
        enable_ask:   是否暴露 ask (反问).
        enable_reuse: 是否暴露 reuse (不检索直接答). 默认 True;
            会话非常初期 (无 history 也无 last_context) 也允许 chitchat/oos 走 reuse.
    """
    lim = limits or DEFAULT_ROUTING_LIMITS
    tools = [build_plan_tool(limits=lim)]
    if enable_multi:
        tools.append(build_multi_tool(limits=lim))
    if enable_ask:
        tools.append(ASK_TOOL)
    if enable_reuse:
        tools.append(REUSE_TOOL)
    return tools


def reflect_tools(*, limits: Optional[RoutingLimits] = None) -> List[Dict[str, Any]]:
    """返回 reflect 的 3 个工具 (ok/retry/partial), 始终全部暴露。"""
    lim = limits or DEFAULT_ROUTING_LIMITS
    return [OK_TOOL, build_retry_tool(limits=lim), PARTIAL_TOOL]


# ---------------------------------------------------------------------------
# 工具名常量 (避免散落字符串)
# ---------------------------------------------------------------------------

TOOL_PLAN = "plan"
TOOL_MULTI = "multi"
TOOL_ASK = "ask"
TOOL_REUSE = "reuse"
TOOL_OK = "ok"
TOOL_RETRY = "retry"
TOOL_PARTIAL = "partial"

ROUTER_TOOL_NAMES = (TOOL_PLAN, TOOL_MULTI, TOOL_ASK, TOOL_REUSE)
REFLECT_TOOL_NAMES = (TOOL_OK, TOOL_RETRY, TOOL_PARTIAL)
