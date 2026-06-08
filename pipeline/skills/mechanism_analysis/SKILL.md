---
id: mechanism_analysis
name: 机理分析
priority: 55
triggers:
  - 机理
  - 机制
  - 原理
  - 为什么
  - 为何
  - 怎么形成
  - 如何形成
  - 成因
  - 作用机制
  - 影响机制
  - mechanism
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

当用户想弄清楚**某个现象/性能背后的机理、原因、作用机制**时选择本 skill。
特征：聚焦单一对象，追问"为什么/怎么发生/由什么决定"，需要沿因果链把过程、影响因素、产物、结果讲清楚，而不是面上综述或多对象对比。
例如："耐候钢的耐蚀机理是什么"、"稀土元素为什么能提升耐蚀性"、"锈层是怎么形成的"。
