---
id: comparison
name: 对比分析
priority: 60
sufficiency:
  min_docs: 2
  need_conflict_check: true
  need_quantitative_data: false
prefer_first_paths:
  - summary
tuning:
  max_rounds: 5
  max_batches: 3
  gap_stall_limit: 2
guards:
  - per_object_evidence
---

## Description

在两个或多个明确对象之间做对比、找区别、评优劣：把多个对象摆在若干共同维度上逐一比较。

## When to use

用户问题里有**明确的比较关系**（A 与 B、A 相比 B、谁更好、优缺点对比等），核心诉求是"在若干维度上把多个对象摆在一起比较"，且被比较的对象是可枚举的具体对象（材料/方法/工艺/方案等）。

## When not to use

- 只问单一对象"是什么/为什么/怎么形成" → 用机理分析。
- 想了解一个主题的整体面貌或研究现状，没有明确的对比对象 → 用文献综述。
- 想看一个方向随时间如何演进 → 用研究进展梳理。
- 只要某个具体数值/速率/含量 → 走通用（或定量抽取）。

## Examples

- 锌铝镁镀层和纯锌镀层的耐蚀性对比
- 低温和高温合成 MoS2 有什么区别
- 几种表征手段各自的优缺点
- PVD 与 CVD 镀膜谁更适合大尺寸基材

## Anti-examples

- 耐候钢的耐蚀机理是什么（单一对象机理 → mechanism_analysis）
- 综述一下锌铝镁镀层的研究现状（主题全貌 → literature_review）
- 固态电解质这些年的发展进展（时间演进 → research_timeline）
