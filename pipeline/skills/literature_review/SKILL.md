---
id: literature_review
name: 文献综述
priority: 30
sufficiency:
  min_docs: 4
  need_conflict_check: false
  need_quantitative_data: false
prefer_first_paths:
  - summary
tuning:
  max_rounds: 4
  max_batches: 3
  gap_stall_limit: 2
guards: []
---

## Description

对一个主题做面上的系统梳理：横向铺开它有哪些子方向、方法、影响因素与代表性结论，给出整体认识。

## When to use

用户想**全面了解一个主题的整体面貌**——它包含哪些方面/方法/方向，目标是"把一个主题横向铺开讲清楚"。问题范围较宽、没有明确的对比对象、不聚焦单一机理、也不强调时间线。

## When not to use

- 强调"这些年怎么发展过来的 / 演进脉络 / 趋势" → 用研究进展梳理（纵向时间线，本 skill 是横向铺开）。
- 在明确的几个对象间评优劣 → 用对比分析。
- 追问单一现象的成因/作用机制 → 用机理分析。

## Examples

- 综述一下锌铝镁镀层的研究现状
- 耐候钢都有哪些研究方向
- 介绍一下这个材料体系的整体情况
- 钢铁表面防腐都有哪些技术路线

## Anti-examples

- MoS2 制备方法的演进历程（按时间演进 → research_timeline）
- 锌铝镁和纯锌镀层耐蚀性对比（明确对比对象 → comparison）
- 稀土为什么能提升耐蚀性（单一机理 → mechanism_analysis）
