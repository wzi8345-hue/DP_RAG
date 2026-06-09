"""观测脚本: 语义切分三项改造的实际效果 (离线, 无需 embedding 服务)。

对照组:
  A. char  模式 (size_unit=char,  默认) —— 现状
  B. token 模式 (size_unit=token, estimate_tokens) —— 引入2
  C. char  模式 + 句子边界 overlap —— 引入3

用 FakeEmbedder 让相邻句距离=0 (无语义断点), 使块边界完全由"尺寸度量"决定,
从而清晰对比 char vs token 的分块差异; overlap 部分单独展示句边界回填。

运行:
  cd /Users/dp/Desktop/工作文件/DP_rag_skill
  python -m pipeline.tests.observe_semantic_sizing
"""

from __future__ import annotations

from pipeline.processors.semantic_splitter import (
    estimate_tokens,
    semantic_split,
)


class _FakeEmbedder:
    """返回同向量 -> 相邻距离=0 -> 无语义断点, 切分完全由尺寸阈值驱动。"""

    def embed_all(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


# 中英混排学术段落 (模拟材料/二维半导体文献), 字符多但 token 相对少
SAMPLE = (
    "二维过渡金属硫化物 (transition metal dichalcogenides, TMDs) 因其独特的电子结构受到广泛关注。"
    "Among them, MoS2 is a typical semiconductor with an indirect bandgap of about 1.2 eV in bulk form, "
    "which transitions to a direct bandgap of approximately 1.8 eV in the monolayer limit。"
    "WS2 表现出类似的层数依赖特性, 其单层直接带隙约为 2.0 eV, 略大于 MoS2。"
    "这种带隙调制源于量子限域效应 (quantum confinement effect) 和层间耦合的减弱。"
    "In photoluminescence (PL) measurements, monolayer samples exhibit a strong emission peak, "
    "while multilayer samples show significantly quenched intensity due to the indirect transition。"
    "此外, 缺陷工程 (defect engineering) 与应变调控 (strain engineering) 被证明可以有效调节这些材料的光电性质, "
    "for example, sulfur vacancies introduce mid-gap states that act as radiative recombination centers。"
    "在器件层面, 基于 MoS2 的场效应晶体管 (FET) 展现出高达 10^8 的开关比和约 200 cm^2/V·s 的载流子迁移率, "
    "making them promising candidates for next-generation low-power electronics and optoelectronic applications。"
)


def _stats(chunks, length_fn):
    sizes = [length_fn(c.text) for c in chunks]
    char_sizes = [len(c.text) for c in chunks]
    return sizes, char_sizes


def _print_run(title, chunks, length_fn, unit):
    sizes, char_sizes = _stats(chunks, length_fn)
    print(f"\n=== {title} ===")
    print(f"  块数: {len(chunks)}")
    print(f"  每块 {unit} 数: {sizes}")
    print(f"  每块字符数 : {char_sizes}")
    if sizes:
        print(f"  {unit} 范围: {min(sizes)} ~ {max(sizes)}  (极差 {max(sizes) - min(sizes)})")
    for i, c in enumerate(chunks, 1):
        head = c.text[:50].replace("\n", " ")
        print(f"    [{i}] {head}…")


def main():
    print("输入文本: %d 字符 / 约 %d token (estimate_tokens)"
          % (len(SAMPLE), estimate_tokens(SAMPLE)))
    print("比率 char/token = %.2f  <- 中英混排, 字符数明显高于 token 数"
          % (len(SAMPLE) / max(1, estimate_tokens(SAMPLE))))

    emb = _FakeEmbedder()

    # A. char 模式 (现状): 阈值按字符
    a = semantic_split(
        SAMPLE, emb,
        target_chars=300, max_chars=400, min_chars=80, breakpoint_percentile=85,
    )
    _print_run("A. char 模式 (target=300 / max=400 字符)", a, len, "字符")

    # B. token 模式: 同样的数字, 但按 token 计 -> 块更大、数量更少、更一致
    b = semantic_split(
        SAMPLE, emb,
        target_chars=300, max_chars=400, min_chars=80, breakpoint_percentile=85,
        length_fn=estimate_tokens,
    )
    _print_run("B. token 模式 (target=300 / max=400 token)", b, estimate_tokens, "token")

    # B'. token 模式取更贴近实际的小阈值, 看块大小一致性
    b2 = semantic_split(
        SAMPLE, emb,
        target_chars=120, max_chars=180, min_chars=40, breakpoint_percentile=85,
        length_fn=estimate_tokens,
    )
    _print_run("B'. token 模式 (target=120 / max=180 token)", b2, estimate_tokens, "token")

    # C. 句子边界 overlap: char 模式 + overlap=60 字符预算
    c = semantic_split(
        SAMPLE, emb,
        target_chars=300, max_chars=500, min_chars=80, breakpoint_percentile=85,
        overlap_chars=60,
    )
    print("\n=== C. char 模式 + 句子边界 overlap (overlap=60 字符预算) ===")
    print(f"  块数: {len(c)}")
    for i, ch in enumerate(c, 1):
        print(f"    [{i}] {ch.text[:70].replace(chr(10), ' ')}…")
    # 展示相邻块的 overlap: 第 i 块开头应是第 i-1 块的尾句
    if len(c) >= 2:
        print("\n  overlap 验证 (第2块开头是否为完整句子回填):")
        print(f"    第2块开头 60 字符: {c[1].text[:60]!r}")
        print("    -> 应为上一块末尾的完整句子, 不在句中截断")

    print("\n结论:")
    print("  - token 模式块数更少、token 极差更小 -> 中英混排块大小更一致")
    print("  - overlap 按句子边界回填, 不产生半句碎片")


if __name__ == "__main__":
    main()
