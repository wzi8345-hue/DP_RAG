# 专家模式 Skills（文件定义式研究技能）

专业研究模式（professional）的"思考方式"做成了**文件定义式 skill**：每个 skill 是一个文件夹，
描述一类研究任务该怎么**规划（plan）**、怎么**逐轮决策下一步检索（policy）**、以及怎么**组织综述输出（synthesis）**。
检索/精排流程对所有 skill 完全一致、不随 skill 改变。

新增一个 skill = 新建一个文件夹放进本目录（或配置的用户目录），**无需改任何 Python 代码**。

## 关键：未命中时回退现有逻辑

`skill_router` 用**思考模型**根据用户发话选 skill：它读取每个 skill 的「适用 / 不适用 / 正例 / 反例」
自描述做语义判断（**不再有触发词关键字匹配**）。若模型判 none 或置信度低于 `min_confidence`，
则 `skill_id = None`，plan / policy / synthesis **逐字使用现有的通用提示词**，行为与未引入 skill 前完全一致。
因此 skill 只做"增强"，从不破坏既有链路。

路由模式（`professional.skills.router.mode`）：`off`=始终走通用；`llm`（默认）=思考模型判断。

## 文件夹结构

```
<skill_id>/
  SKILL.md          # 必需：frontmatter + 自描述段(Description / When to use / When not to use / Examples / Anti-examples)
  plan.md           # 必需：规划提示词（research_plan 选中时的拆解指导，替换通用拆解段）
  policy.md         # 必需：策略提示词（每轮 continue/finish/clarify 的判断指导，替换通用判断段）
  synthesis.md      # 可选：综述输出结构（## System / ## Thinking / ## User 三段；缺失则回退通用模板）
  eval_cases.yaml   # 可选：路由评测用例（should_hit / should_miss），供 eval_router.py 校准，不进运行链路
```

只有 `SKILL.md / plan.md / policy.md / synthesis.md` 会被加载器读取。

## SKILL.md frontmatter 字段

| 字段 | 说明 |
|---|---|
| `id` | skill 唯一标识（与文件夹名一致）|
| `name` | 中文展示名 |
| `priority` | 极少数模型判定平手时数值大者优先（一般无需调）|
| `sufficiency` | 默认收口标准（min_docs / need_conflict_check / need_quantitative_data / must_cover），当规划 LLM 未给出时注入；会被现有 observation 机制展示给 policy |
| `prefer_first_paths` | 首轮偏好的检索路径提示（写进 plan 提示词，仅引导）|
| `tuning` | 覆盖该 skill 的 `max_rounds / max_batches / gap_stall_limit / stall_quality_floor` |
| `guards` | 声明式守卫名（见下），逐轮把"未满足项"注入 policy 观测以引导继续检索 |

## SKILL.md 自描述段（决定路由）

正文用 `##` 小节描述"这个 skill 该不该被选"，思考模型据此判断（标题中英皆可识别）：

| 小节 | 作用 |
|---|---|
| `## Description` | 一句话概述这个 skill 在做什么 |
| `## When to use` | 适用场景：什么样的提问该用它（越具体越准）|
| `## When not to use` | 不适用场景：与相邻 skill 划清边界，避免误命中 |
| `## Examples` | 正例：应命中本 skill 的代表性问题（每行一条）|
| `## Anti-examples` | 反例：看似相关但应归别处的问题（注明该归哪类）|

> 调试路由命中率：`python -m pipeline.skills.eval_router`（依赖真实分类模型，跑各 skill 的 `eval_cases.yaml`）。

## guards（声明式守卫，软引导）

守卫只把"充分性未满足项"追加进 policy 的观测文本，引导 LLM 继续补检（不做硬阻断，
既有的轮次/stall/缺口熔断负责防止空转）。当前内置：

- `min_docs`：证据文献数未达 `sufficiency.min_docs` 时提示。
- `need_quantitative`：证据里没有"数值+单位"时提示（正则判定）。
- `per_object_evidence`：规划出的 facet 维度仍有未被判定覆盖的（对比任务里 = 某个被比较对象还缺证据）→ 列出未覆盖维度，提示对称覆盖后再收口。
- `causal_chain_evidence`：证据里没有因果/机理连接词（因为/导致/机理/mechanism…）→ 提示证据偏现象描述、需补因果链证据。

## plan.md / policy.md 提示词写法

- `plan.md` 的内容会替换通用规划提示词里"选择 research_plan 时…"这一段；外层仍保留
  "先判断 reject/clarify/plan"的三选一闸门与 function-calling 约束，因此 plan.md 只需写
  **选中 research_plan 后该如何拆解 facets 与首轮批次**。
- `policy.md` 的内容会替换通用策略提示词里"判断依据 / 你的任务 / 效率约束"这几段；外层仍保留
  "你会看到哪些观测信息"与 function-calling 约束。
