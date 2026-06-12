---
id: mechanism_analysis
name: 机理分析
priority: 55
sufficiency:
  min_docs: 2
  need_conflict_check: true
  need_quantitative_data: false
prefer_first_paths:
  - progressive
tuning:
  max_rounds: 5
  max_batches: 3
  gap_stall_limit: 2
guards:
  - causal_chain_evidence
---

## Description

弄清某个现象/性能背后的机理与因果：沿因果链把过程、影响因素、产物、结果讲清楚。

## When to use

用户聚焦**单一对象/现象**，追问"为什么 / 怎么发生 / 由什么决定 / 如何形成 / 作用机制是什么"，需要解释原因与因果过程，而不是罗列或比较。

## When not to use

- 在多个对象间比较优劣 → 用对比分析。
- 想要主题全貌或研究现状 → 用文献综述。
- 想看一个方向随时间的演进 → 用研究进展梳理。

## Examples

- 耐候钢的耐蚀机理是什么
- 稀土元素为什么能提升耐蚀性
- 致密锈层是怎么形成的
- 合金元素如何影响镀层的自修复行为

## Anti-examples

- 锌铝镁和纯锌镀层耐蚀性对比（多对象比较 → comparison）
- 综述一下耐候钢的研究现状（主题全貌 → literature_review）
- 耐候钢耐蚀研究这些年的进展（时间演进 → research_timeline）
