---
id: research_timeline
name: 研究进展梳理
priority: 56
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

把一个研究方向沿时间/技术代际串成一条线：早期是什么、关键转折在哪、现在做到哪、往哪走。

## When to use

用户的核心诉求是**纵向的时间演进**——某主题"这些年怎么发展过来的"、演进脉络、发展历程、代表性工作的先后、当前趋势与未来方向。关键在"沿时间/代际推进"，强调每一步相比上一步推进了什么。

## When not to use

- 只想横向了解主题有哪些方面/方法（不强调时间先后）→ 用文献综述。
- 在明确的几个对象间比优劣 → 用对比分析。
- 追问单一机理/成因 → 用机理分析。

边界提示：单说"研究现状"既可能是横向综述、也可能是纵向进展——若问句带"发展/演进/历程/趋势/近年来/这些年"等时间线索，归本 skill；否则更可能是文献综述。

## Examples

- 锌铝镁镀层耐蚀性研究这些年的发展进展
- MoS2 制备方法的演进历程
- 固态电解质近年来的研究趋势与未来方向
- 耐候钢防腐技术是怎么一步步发展到今天的

## Anti-examples

- 综述一下锌铝镁镀层都有哪些研究方向（横向铺开、无时间线 → literature_review）
- 锌铝镁和纯锌镀层耐蚀性对比（明确对比 → comparison）
- 稀土为什么能提升耐蚀性（单一机理 → mechanism_analysis）
