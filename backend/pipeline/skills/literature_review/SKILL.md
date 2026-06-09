---
id: literature_review
name: 文献综述
priority: 30
triggers:
  - 综述
  - 现状
  - 梳理
  - 概述
  - 介绍一下
  - 总体情况
  - 研究进展
  - review
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

当用户希望对某个主题做**面上的系统梳理 / 研究现状综述 / 总体介绍**时选择本 skill。
特征：问题范围较宽、目标是"全面了解一个主题有哪些方面/进展"，而不是聚焦单一对比、单一机理或单一数值。
例如："综述一下锌铝镁镀层的研究现状"、"耐候钢都有哪些研究方向"、"介绍一下这个材料体系"。
