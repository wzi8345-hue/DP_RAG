---
id: comparison
name: 对比分析
priority: 60
triggers:
  - 对比
  - 比较
  - 区别
  - 差异
  - 优劣
  - 优缺点
  - 相比
  - 谁更
  - 哪个更
  - vs
  - versus
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

当用户要在**两个或多个对象之间做对比 / 找区别 / 评优劣**时选择本 skill。
特征：发话里出现明确的比较关系（A 与 B、A 相比 B、谁更好、优缺点对比等），核心诉求是"在若干维度上把多个对象摆在一起比较"。
例如："锌铝镁镀层和纯锌镀层的耐蚀性对比"、"低温和高温合成 MoS2 有什么区别"、"几种表征手段的优劣"。
