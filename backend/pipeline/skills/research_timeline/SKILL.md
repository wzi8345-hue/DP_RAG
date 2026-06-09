---
id: research_timeline
name: 研究进展梳理
priority: 56
triggers:
  - 进展
  - 发展
  - 演进
  - 演化
  - 脉络
  - 历程
  - 趋势
  - 现状
  - 近年来
  - 最新进展
  - 发展历程
  - 研究现状
  - 发展方向
  - roadmap
sufficiency:
  min_docs: 3
  need_conflict_check: false
  need_quantitative_data: false
tuning:
  max_rounds: 5
  max_batches: 3
  gap_stall_limit: 2
guards:
  - min_docs
---

## Description

当用户想了解**某一主题随时间的发展脉络 / 演进过程 / 代表性工作 / 当前趋势**时选择本 skill。
特征：核心诉求不是"对比 A 与 B"也不是"解释某个机理"，而是"把一个研究方向沿时间或技术代际**串成一条线**"——早期是什么、关键转折在哪、现在做到哪、往哪走。
例如："锌铝镁镀层耐蚀性研究这些年的发展进展"、"MoS2 制备方法的演进历程"、"固态电解质近年来的研究现状与趋势"。
与「文献综述」的区别：综述按维度横向铺开，本 skill 强调**时间/代际纵向演进**与"每一步相比上一步推进了什么"。
