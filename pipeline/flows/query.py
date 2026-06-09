"""Query 流程: 检索 → 生成, 直接输出模型结果。

支持:
1. 简单检索 (vector/metadata/hybrid) + 基础生成
2. Agentic RAG 检索 + 多路径生成
3. LangGraph Agentic RAG + 自我反思循环
4. 流式/非流式输出
5. Pydantic 验证的输出结果
6. 多轮对话 (ChatSession)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..clients.client_registry import get_global_registry
from ..clients.llm import LLMClient
from ..clients.milvus import resolve_milvus_connection
from ..clients.query_format import instruct_kwargs_from_embedding_cfg
from ..retrieval.retrievers import (
    build_retrievers,
    _build_filter_expr,
    Hit,
    MetadataRetriever,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
)
from ..retrieval.context_builder import ContextBuilder
from ..retrieval.agentic import (
    AgenticRAGPipeline,
    build_agentic_pipeline,
    DEFAULT_AGENTIC_SYSTEM_PROMPT,
    AGENTIC_USER_TEMPLATE,
)
from ..retrieval.langgraph_agent import REUSE_USER_TEMPLATE
from ..retrieval.progressive_config import progressive_config_from_dict, summary_config_from_dict
from ..retrieval.hybrid_config import hybrid_config_from_dict
from ..retrieval.hybrid_weights import (
    STAGE_SIMPLE,
    infer_hybrid_weights,
    infer_retrieve_bias_heuristic,
    format_weight_log,
)
from ..models import QueryResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 多轮对话数据结构
# ---------------------------------------------------------------------------

# 历史中只保留用户发话 + 模型最终回复; 不挂检索决策、不挂检索 context (避免 token 爆炸)
DEFAULT_MAX_HISTORY_TURNS = 5


@dataclass
class ChatTurn:
    """一轮对话记录: 仅 user query + assistant answer。"""
    query: str
    answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """多轮对话会话, 持有最近 N 轮 user/assistant 文本。

    设计原则:
    - 只保留 user 发话和 LLM 最终回复, 不再挂检索 decision/context
      (检索每轮都重新跑, 历史 context 不需要保留, 否则 prompt token 指数膨胀)
    - 通过 max_turns 截断, 默认仅保留最近 5 轮
    """
    turns: List[ChatTurn] = field(default_factory=list)
    max_turns: int = DEFAULT_MAX_HISTORY_TURNS

    def add_turn(self, query: str, answer: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.turns.append(ChatTurn(query=query, answer=answer, meta=meta or {}))
        # 写入即截断, 内存与 token 双重稳定
        if self.max_turns and len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def recent_messages(self) -> List[Dict[str, str]]:
        """返回截断后的 user/assistant message 列表 (不含 system 与当前 user)。"""
        recent = self.turns[-self.max_turns:] if self.max_turns else self.turns
        msgs: List[Dict[str, str]] = []
        for t in recent:
            msgs.append({"role": "user", "content": t.query})
            msgs.append({"role": "assistant", "content": t.answer})
        return msgs

    def to_messages(self, system: str, current_user: str) -> List[Dict[str, str]]:
        """构建 OpenAI messages 列表 (含历史 + 当前 user)。"""
        return (
            [{"role": "system", "content": system}]
            + self.recent_messages()
            + [{"role": "user", "content": current_user}]
        )

    def last_turn_meta(self) -> Dict[str, Any]:
        if not self.turns:
            return {}
        return self.turns[-1].meta or {}


# ---------------------------------------------------------------------------
# QueryFlow
# ---------------------------------------------------------------------------

class QueryFlow:
    """检索 + 生成的完整查询流程。

    用法:
        flow = QueryFlow(config)
        result = flow.run("MoS2 的晶格常数是多少?")
        print(result.answer)

        # 多轮对话
        session = ChatSession()
        result, session = flow.run("MoS2 的晶格常数?", session=session)
        result, session = flow.run("那它的带隙呢?", session=session)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # 懒加载缓存: 复用底层连接 / 客户端, 避免每次 query 都重建
        self._agentic_pipeline: Optional[AgenticRAGPipeline] = None
        self._langgraph_agent: Optional[Any] = None
        self._research_agent: Optional[Any] = None
        self._simple_retrievers: Optional[
            Tuple[MetadataRetriever, VectorRetriever, BM25Retriever, HybridRetriever]
        ] = None
        self._simple_llm: Optional[LLMClient] = None

    # ── 懒加载工厂方法 ────────────────────────────────────────────────

    def invalidate_caches(self) -> None:
        """清空所有懒加载的 agent/retriever 缓存, 使下次访问时以当前
        config.milvus['collection'] 重新构建。用于切换目标集合时调用。"""
        self._agentic_pipeline = None
        self._langgraph_agent = None
        self._research_agent = None
        self._simple_retrievers = None
        logger.info("[QueryFlow] caches invalidated (collection may have changed)")

    def reload_skills(self) -> None:
        """仅丢弃研究 agent 缓存, 使下次专业模式请求重新加载 skill 目录
        (上传/编辑/删除 skill 后调用; 不重建昂贵的 agentic pipeline)。"""
        self._research_agent = None
        logger.info("[QueryFlow] research agent 缓存已清, 下次专业模式将重载 skills")

    def _get_agentic_pipeline(self) -> AgenticRAGPipeline:
        if self._agentic_pipeline is None:
            gen_cfg = self.config.generation
            milvus_cfg = self.config.milvus
            emb_cfg = self.config.embedding
            ret_cfg = self.config.retrieval
            router_cfg = gen_cfg.get("router", {}) or {}
            search_cfg = ret_cfg.get("search", {}) or {}
            index_cfg = milvus_cfg.get("index", {}) or {}
            ctx_cfg = ret_cfg.get("context", {}) or {}
            milvus_uri, milvus_token, milvus_db = resolve_milvus_connection(milvus_cfg)
            self._agentic_pipeline = build_agentic_pipeline(
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                db_name=milvus_db,
                collection=milvus_cfg.get("collection", "literature_chunks"),
                embed_api_base=emb_cfg.get("api_base", ""),
                embed_model=emb_cfg.get("model", "model"),
                embed_api_key=emb_cfg.get("api_key", ""),
                llm_api_base=gen_cfg.get("api_base", "https://api.gpugeek.com/v1"),
                llm_model=gen_cfg.get("model", "DeepSeek/DeepSeek-V3-0324"),
                llm_api_key=gen_cfg.get("api_key", ""),
                # router LLM 独立配置 (None 即复用生成 LLM)
                router_api_base=router_cfg.get("api_base"),
                router_model=router_cfg.get("model"),
                router_api_key=router_cfg.get("api_key"),
                router_temperature=float(router_cfg.get("temperature", 0.0)),
                router_max_tokens=int(router_cfg.get("max_tokens", 300)),
                router_timeout=int(router_cfg.get("timeout", 30)),
                router_max_retries=int(router_cfg.get("max_retries", 1)),
                router_history_turns=int(router_cfg.get("history_turns", 1)),
                router_use_json_schema=bool(router_cfg.get("use_json_schema", False)),
                use_router_llm=True,
                enable_generation_llm=True,
                embed_normalize=bool(emb_cfg.get("normalize", False)),
                dense_weight=float(ret_cfg.get("dense_weight", 0.6)),
                bm25_weight=float(ret_cfg.get("bm25_weight", 0.4)),
                dense_metric=str(index_cfg.get("dense_metric", "IP")),
                dense_search_params=search_cfg.get("dense") or None,
                bm25_search_params=search_cfg.get("bm25") or None,
                **instruct_kwargs_from_embedding_cfg(emb_cfg),
                context_max_total_chars=ctx_cfg.get("max_total_chars"),
                context_per_route_chars=ctx_cfg.get("per_route_chars"),
                context_max_in_prompt=ctx_cfg.get("max_in_prompt"),
                progressive_config=progressive_config_from_dict(
                    ret_cfg.get("progressive"),
                ),
                summary_config=summary_config_from_dict(
                    ret_cfg.get("summary"),
                ),
                hybrid_config=hybrid_config_from_dict(
                    ret_cfg.get("hybrid"),
                ),
                keepalive_time_ms=int(milvus_cfg.get("keepalive_time_ms", 300_000)),
                keepalive_timeout_ms=int(milvus_cfg.get("keepalive_timeout_ms", 60_000)),
                disable_thinking=bool(gen_cfg.get("disable_thinking", True)),
                disable_thinking_extra_body=bool(gen_cfg.get("disable_thinking_extra_body", False)),
            )
        return self._agentic_pipeline

    def _get_simple_retrievers(
        self,
    ) -> Tuple[MetadataRetriever, VectorRetriever, BM25Retriever, HybridRetriever]:
        if self._simple_retrievers is None:
            milvus_cfg = self.config.milvus
            emb_cfg = self.config.embedding
            ret_cfg = self.config.retrieval
            search_cfg = ret_cfg.get("search", {}) or {}
            index_cfg = milvus_cfg.get("index", {}) or {}
            milvus_uri, milvus_token, milvus_db = resolve_milvus_connection(milvus_cfg)
            self._simple_retrievers = build_retrievers(
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                db_name=milvus_db,
                collection=milvus_cfg.get("collection", "literature_chunks"),
                embed_api_base=emb_cfg.get("api_base", ""),
                embed_model=emb_cfg.get("model", "model"),
                embed_api_key=emb_cfg.get("api_key", ""),
                embed_normalize=bool(emb_cfg.get("normalize", False)),
                dense_weight=float(ret_cfg.get("dense_weight", 0.6)),
                bm25_weight=float(ret_cfg.get("bm25_weight", 0.4)),
                dense_metric=str(index_cfg.get("dense_metric", "IP")),
                dense_search_params=search_cfg.get("dense") or None,
                bm25_search_params=search_cfg.get("bm25") or None,
                **instruct_kwargs_from_embedding_cfg(emb_cfg),
            )
        return self._simple_retrievers

    def _get_simple_llm(self) -> LLMClient:
        if self._simple_llm is None:
            gen_cfg = self.config.generation
            self._simple_llm = get_global_registry().get_llm(
                api_base=gen_cfg.get("api_base", "https://api.gpugeek.com/v1"),
                model=gen_cfg.get("model", "DeepSeek/DeepSeek-V3-0324"),
                api_key=gen_cfg.get("api_key", ""),
                timeout=gen_cfg.get("timeout", 120),
                max_retries=gen_cfg.get("max_retries", 3),
                disable_thinking_extra_body=bool(gen_cfg.get("disable_thinking_extra_body", False)),
            )
        return self._simple_llm

    def run(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        use_agentic: bool = True,
        output_file: Optional[str] = None,
        session: Optional[ChatSession] = None,
        professional: bool = False,
    ) -> Tuple[QueryResult, ChatSession]:
        """执行查询流程: 检索 → 生成。

        Args:
            query: 用户问题
            mode: 检索模式 (hybrid/vector/metadata), 仅非 agentic 模式
            top_k: 返回 top_k 条结果
            stream: 是否流式输出
            use_agentic: 是否使用 Agentic RAG (默认 True)
            output_file: 可选结果输出文件路径
            session: 多轮对话会话 (None 则新建)

        Returns:
            (QueryResult, ChatSession) 元组
        """
        if session is None:
            gen_cfg = self.config.generation
            max_turns = int(gen_cfg.get("max_history_turns", DEFAULT_MAX_HISTORY_TURNS))
            session = ChatSession(max_turns=max_turns)

        # 历史: 只用截断后的 user/assistant 文本, 不再嵌入旧 context
        history = session.recent_messages() or None

        # 检查 LangGraph 开关
        lg_cfg = self.config.retrieval.get("langgraph", {}) or {}
        use_langgraph = use_agentic and bool(lg_cfg.get("enabled", False))

        prof_cfg = lg_cfg.get("professional", {}) or {}
        use_professional = (
            professional and use_langgraph and bool(prof_cfg.get("enabled", True))
        )

        if use_professional:
            result = self._run_research(
                query, stream=stream, output_file=output_file,
                history=history, session=session,
            )
        elif use_langgraph:
            result = self._run_langgraph(
                query, stream=stream, output_file=output_file,
                history=history, session=session,
            )
        elif use_agentic:
            result = self._run_agentic(
                query, stream=stream, output_file=output_file,
                history=history, session=session,
            )
        else:
            result = self._run_simple(
                query, mode=mode, top_k=top_k, stream=stream,
                output_file=output_file, history=history, session=session,
            )

        session.add_turn(query=query, answer=result.answer, meta=result.session_meta)
        return result, session

    # ── LangGraph agent ──────────────────────────────────────────────────

    def _get_langgraph_agent(self):
        """懒加载 LangGraph agent: 复用 _get_agentic_pipeline() 的所有组件。"""
        if self._langgraph_agent is None:
            from ..retrieval.langgraph_agent import build_langgraph_agent_from_pipeline
            from ..clients.reranker import RerankerClient

            pipeline = self._get_agentic_pipeline()
            ret_cfg = self.config.retrieval
            lg_cfg = ret_cfg.get("langgraph", {}) or {}
            ref_cfg = lg_cfg.get("reflection", {}) or {}
            rerank_cfg = lg_cfg.get("reranker", {}) or {}
            gen_cfg = self.config.generation

            # 构建 reflection LLM
            # disable_thinking_extra_body 决定是否走 vLLM 专属的 chat_template_kwargs:
            #   - reflection 配置里显式给了就用反思自己的设置
            #   - 否则回退到 generation 的设置 (默认 False, 即云平台模式)
            r_extra_body_cfg = ref_cfg.get("disable_thinking_extra_body")
            if r_extra_body_cfg is None:
                r_extra_body_cfg = gen_cfg.get("disable_thinking_extra_body", False)
            r_extra_body = bool(r_extra_body_cfg)

            reflection_llm = None
            if ref_cfg.get("enabled", True):
                r_api_base = ref_cfg.get("api_base") or gen_cfg.get("api_base")
                r_model = ref_cfg.get("model") or gen_cfg.get("model")
                r_api_key = ref_cfg.get("api_key") or gen_cfg.get("api_key")
                if r_api_key:
                    try:
                        reflection_llm = LLMClient(
                            api_base=r_api_base, model=r_model, api_key=r_api_key,
                            timeout=int(ref_cfg.get("timeout", 15)),
                            max_retries=1,
                            disable_thinking_extra_body=r_extra_body,
                        )
                        logger.info(
                            f"[langgraph] reflection LLM: model={r_model} "
                            f"timeout={ref_cfg.get('timeout', 15)}s "
                            f"extra_body={r_extra_body} "
                            f"({'vLLM' if r_extra_body else '云平台'})"
                        )
                    except Exception as e:
                        logger.warning(f"[langgraph] reflection LLM 初始化失败: {e}")

            max_retries = int(lg_cfg.get("max_retries", 2))
            if not ref_cfg.get("enabled", True):
                max_retries = 0

            # 构建 reranker client (远程 API 模式)
            reranker_client = None
            if rerank_cfg.get("enabled", False):
                try:
                    reranker_client = RerankerClient(
                        api_base=rerank_cfg.get("api_base", "http://localhost:8001/v1"),
                        model=rerank_cfg.get("model", "Qwen/Qwen3-Reranker-0.6B"),
                        api_key=rerank_cfg.get("api_key") or gen_cfg.get("api_key", ""),
                        top_k=int(rerank_cfg.get("top_k", 5)),
                        score_threshold=float(rerank_cfg.get("quality_threshold", 0.5)),
                        timeout=int(rerank_cfg.get("timeout", 60)),
                        max_retries=int(rerank_cfg.get("max_retries", 2)),
                        # P1 #17: 默认 fail_open, API 故障时降级为不 rerank
                        fail_open=bool(rerank_cfg.get("fail_open", True)),
                    )
                    logger.info(
                        f"[langgraph] reranker: api_base={rerank_cfg.get('api_base', 'http://localhost:8001/v1')} "
                        f"model={rerank_cfg.get('model', 'Qwen/Qwen3-Reranker-0.6B')} "
                        f"top_k={rerank_cfg.get('top_k', 5)} "
                        f"quality_threshold={rerank_cfg.get('quality_threshold', 0.5)} "
                        f"fail_open={rerank_cfg.get('fail_open', True)}"
                    )
                except Exception as e:
                    logger.warning(f"[langgraph] reranker 初始化失败, 禁用: {e}")

            # 反思 LLM 的 disable_thinking 仅在 vLLM 模式下下发:
            #   - r_extra_body=True (vLLM): 取 reflection 自己的 disable_thinking, 缺省
            #     回退到 generation 配置; 通过 chat_template_kwargs.enable_thinking 控制
            #   - r_extra_body=False (云平台): 传 None, 不下发任何 thinking 控制参数,
            #     既不发 chat_template_kwargs, 也不追加 /no_think 文本
            if r_extra_body:
                reflect_dt_cfg = ref_cfg.get("disable_thinking")
                if reflect_dt_cfg is None:
                    reflect_dt_cfg = gen_cfg.get("disable_thinking", True)
                reflect_disable_thinking: Optional[bool] = bool(reflect_dt_cfg)
            else:
                reflect_disable_thinking = None

            # ── Function Calling 路由 (v4): 按 config 开关注入 routing_core ──
            routing_core = self._maybe_build_routing_core(
                pipeline, reflection_llm, reflect_disable_thinking,
            )

            from ..retrieval.rerank_diagnosis import RerankDiagnosisConfig
            diag_raw = rerank_cfg.get("diagnosis") or {}
            skip_causes_raw = diag_raw.get("skip_reflect_causes")
            if skip_causes_raw is None:
                skip_reflect_causes = ("wrong_type", "wrong_route")
            else:
                skip_reflect_causes = tuple(str(c) for c in skip_causes_raw)
            # P1 #9: 默认 skip_reflect_confidence 提到 0.90, 与 wrong_type strong (0.92) 拉开距离
            rerank_diagnosis_config = RerankDiagnosisConfig(
                enabled=bool(diag_raw.get("enabled", True)),
                skip_reflect_confidence=float(
                    diag_raw.get("skip_reflect_confidence", 0.90),
                ),
                skip_reflect_causes=skip_reflect_causes,
                type_low_ratio=float(diag_raw.get("type_low_ratio", 0.5)),
                route_dead_score=float(diag_raw.get("route_dead_score", 0.15)),
                narrow_hit_cap=int(diag_raw.get("narrow_hit_cap", 3)),
                broad_hit_floor=int(diag_raw.get("broad_hit_floor", 15)),
                wrong_type_strong_confidence=float(
                    diag_raw.get("wrong_type_strong_confidence", 0.92),
                ),
                wrong_type_weak_confidence=float(
                    diag_raw.get("wrong_type_weak_confidence", 0.80),
                ),
                wrong_type_refs_confidence=float(
                    diag_raw.get("wrong_type_refs_confidence", 0.86),
                ),
                wrong_route_confidence=float(
                    diag_raw.get("wrong_route_confidence", 0.86),
                ),
                too_narrow_confidence=float(
                    diag_raw.get("too_narrow_confidence", 0.72),
                ),
                too_narrow_relax_confidence=float(
                    diag_raw.get("too_narrow_relax_confidence", 0.70),
                ),
                too_broad_confidence=float(
                    diag_raw.get("too_broad_confidence", 0.75),
                ),
                zero_confidence=float(
                    diag_raw.get("zero_confidence", 0.35),
                ),
            )

            # P1 #5: chunk_type-aware 阈值表 (旧字段, 仅 by_type 维度)
            qt_by_type_raw = rerank_cfg.get("quality_threshold_by_type") or {}
            reranker_quality_threshold_by_type: Optional[Dict[str, float]] = None
            if qt_by_type_raw:
                reranker_quality_threshold_by_type = {
                    str(k).lower(): float(v) for k, v in qt_by_type_raw.items()
                }

            # P1.1 (2026-05): per-(route, stage, type) 阈值矩阵
            # 新字段 `quality_thresholds.by_route` 优先, 老字段作为 fallback
            from ..retrieval.quality_thresholds import RouteThresholds
            qt_matrix_raw = rerank_cfg.get("quality_thresholds")
            reranker_route_thresholds = RouteThresholds.from_dict(
                qt_matrix_raw,
                legacy_default=float(rerank_cfg.get("quality_threshold", 0.25)),
                legacy_by_type=reranker_quality_threshold_by_type,
            )

            # P2.3: fail-open 时的 emb_score 最低质量门 (None=禁用安全网, 与旧行为一致)
            fail_open_min_emb_quality = rerank_cfg.get("fail_open_min_emb_quality")
            if fail_open_min_emb_quality is not None:
                fail_open_min_emb_quality = float(fail_open_min_emb_quality)

            from ..retrieval.reflect_summary import ReflectSummaryConfig

            summary_raw = ref_cfg.get("summary") or {}
            reflect_summary_config = ReflectSummaryConfig(
                snippet_chars=int(summary_raw.get("snippet_chars", 400)),
                max_hits_per_route=int(summary_raw.get("max_hits_per_route", 6)),
                max_total_chars=int(summary_raw.get("max_total_chars", 5000)),
                max_chars_per_route=int(summary_raw.get("max_chars_per_route", 2000)),
            )

            summary_ret_cfg = summary_config_from_dict(ret_cfg.get("summary"))

            self._langgraph_agent = build_langgraph_agent_from_pipeline(
                pipeline,
                reflection_llm=reflection_llm,
                max_retries=max_retries,
                reflection_temperature=float(ref_cfg.get("temperature", 0.0)),
                reflection_max_tokens=int(ref_cfg.get("max_tokens", 200)),
                reranker_client=reranker_client,
                reranker_top_k=int(rerank_cfg.get("top_k", 5)),
                reranker_quality_k=int(rerank_cfg.get("quality_k", 3)),
                reranker_quality_threshold=float(rerank_cfg.get("quality_threshold", 0.5)),
                reranker_quality_threshold_by_type=reranker_quality_threshold_by_type,
                reranker_route_thresholds=reranker_route_thresholds,
                fail_open_min_emb_quality=fail_open_min_emb_quality,
                reranker_diagnosis_config=rerank_diagnosis_config,
                disable_thinking=reflect_disable_thinking,
                routing_core=routing_core,
                reflect_summary_config=reflect_summary_config,
                summary_top_docs=summary_ret_cfg.top_docs,
                summary_per_query_k=summary_ret_cfg.per_query_k,
            )
        return self._langgraph_agent

    def _maybe_build_routing_core(
        self,
        pipeline,
        reflection_llm: Optional[LLMClient],
        reflect_disable_thinking: Optional[bool],
    ):
        """按 config.retrieval.langgraph.routing 决定是否构造 RoutingCore (FC 模式)。

        config 节示例:
            retrieval:
              langgraph:
                routing:
                  mode: "fc"                       # legacy | fc; 默认 legacy
                  enable_multi: true
                  enable_ask: false
                  parallel_tool_calls: false
                  history_turns: 1
                  # ── 思考开关 (隐式 CoT) ──
                  # null=不下发 (云平台默认)
                  # true=关闭思考 (FC 隐式 CoT 不推荐, 路由质量会下降)
                  # false=显式开启 (推荐, 但 vLLM 启动需配 --reasoning-parser deepseek_r1)
                  router_disable_thinking: false   # 单独控制 router LLM
                  reflect_disable_thinking: false  # 单独控制 reflect LLM
                  # 未设细粒度开关时, 回退到 reflection.disable_thinking (向后兼容)

        思考开关优先级:
          routing.router_disable_thinking / routing.reflect_disable_thinking  (FC 模式专用, 推荐)
          → reflection.disable_thinking                                       (向后兼容, 仅 reflect 端)
          → null (不下发任何 thinking 控制参数)
        """
        lg_cfg = self.config.retrieval.get("langgraph", {}) or {}
        routing_cfg = lg_cfg.get("routing", {}) or {}
        mode = str(routing_cfg.get("mode", "legacy")).lower()
        if mode != "fc":
            logger.info(f"[langgraph] routing.mode={mode!r}, 走 legacy router")
            return None

        try:
            from ..routing import build_routing_core_from_query_router
            from ..routing.limits import RoutingLimits
        except ImportError as e:
            logger.warning(
                f"[langgraph] routing.mode=fc 但 routing 模块不可用 ({e}), 降级 legacy"
            )
            return None

        # ── 思考开关解析 ──
        # 1. 优先读 routing 节里的细粒度开关 (router/reflect 各自配置)
        # 2. 缺省时分别回退:
        #    - router 端: 既没 generation 也没 reflection 可继承, 默认 None (云平台行为)
        #    - reflect 端: 回退到 reflection.disable_thinking (即旧的 reflect_disable_thinking)
        # 3. 显式 null 视为"不下发"
        router_dt_cfg = routing_cfg.get("router_disable_thinking", "__UNSET__")
        reflect_dt_cfg = routing_cfg.get("reflect_disable_thinking", "__UNSET__")

        router_disable_thinking: Optional[bool]
        if router_dt_cfg == "__UNSET__":
            router_disable_thinking = None
        elif router_dt_cfg is None:
            router_disable_thinking = None
        else:
            router_disable_thinking = bool(router_dt_cfg)

        effective_reflect_dt: Optional[bool]
        if reflect_dt_cfg == "__UNSET__":
            effective_reflect_dt = reflect_disable_thinking  # 向后兼容: 沿用 reflection.disable_thinking
        elif reflect_dt_cfg is None:
            effective_reflect_dt = None
        else:
            effective_reflect_dt = bool(reflect_dt_cfg)

        # ── max_tokens 解析 ──
        # router_max_tokens 优先级:
        #   routing.router_max_tokens > generation.router.max_tokens > 内置默认 600
        # reflect_max_tokens 优先级:
        #   routing.reflect_max_tokens > reflection.max_tokens > 内置默认 500
        gen_router_cfg = (self.config.generation.get("router", {}) or {})
        ref_cfg_local = (lg_cfg.get("reflection", {}) or {})
        router_mt_cfg = routing_cfg.get("router_max_tokens")
        reflect_mt_cfg = routing_cfg.get("reflect_max_tokens")

        router_max_tokens: Optional[int]
        if router_mt_cfg is not None:
            router_max_tokens = int(router_mt_cfg)
        elif gen_router_cfg.get("max_tokens") is not None:
            router_max_tokens = int(gen_router_cfg["max_tokens"])
        else:
            router_max_tokens = None  # 走 RoutingCore 内置默认

        reflect_max_tokens: Optional[int]
        if reflect_mt_cfg is not None:
            reflect_max_tokens = int(reflect_mt_cfg)
        elif ref_cfg_local.get("max_tokens") is not None:
            reflect_max_tokens = int(ref_cfg_local["max_tokens"])
        else:
            reflect_max_tokens = None  # 走 RoutingCore 内置默认

        try:
            routing_limits = RoutingLimits(
                max_paths_per_sub=int(routing_cfg.get("max_paths_per_sub", 2)),
                max_subqueries=int(routing_cfg.get("max_subqueries", 3)),
            )
            core = build_routing_core_from_query_router(
                query_router=pipeline.router,
                reflect_llm=reflection_llm,
                enable_multi=bool(routing_cfg.get("enable_multi", True)),
                enable_ask=bool(routing_cfg.get("enable_ask", False)),
                enable_reuse=bool(routing_cfg.get("enable_reuse", True)),
                router_disable_thinking=router_disable_thinking,
                reflect_disable_thinking=effective_reflect_dt,
                router_max_tokens=router_max_tokens,
                reflect_max_tokens=reflect_max_tokens,
                parallel_tool_calls=routing_cfg.get("parallel_tool_calls", False),
                history_turns=int(routing_cfg.get("history_turns", 1)),
                routing_limits=routing_limits,
            )
            logger.info(
                f"[langgraph] routing.mode=fc enabled: "
                f"enable_multi={core.enable_multi} enable_ask={core.enable_ask} "
                f"enable_reuse={core.enable_reuse} "
                f"parallel_tool_calls={core.parallel_tool_calls} "
                f"max_paths_per_sub={core.routing_limits.max_paths_per_sub} "
                f"max_subqueries={core.routing_limits.max_subqueries} "
                f"router_thinking={_describe_thinking(core.router_disable_thinking)} "
                f"reflect_thinking={_describe_thinking(core.reflect_disable_thinking)} "
                f"router_max_tokens={core.router_max_tokens} "
                f"reflect_max_tokens={core.reflect_max_tokens} "
                f"reflect_llm={'on' if reflection_llm else 'off'}"
            )
            return core
        except Exception as e:
            logger.warning(
                f"[langgraph] RoutingCore 构造失败 ({type(e).__name__}: {e}), 降级 legacy"
            )
            return None

    def _run_langgraph(
        self,
        query: str,
        stream: bool = False,
        output_file: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session: Optional[ChatSession] = None,
    ) -> QueryResult:
        """LangGraph agent 模式: 自我反思循环 + 检索 + 生成。"""
        gen_cfg = self.config.generation
        agent = self._get_langgraph_agent()

        system_prompt = gen_cfg.get("system_prompt") or DEFAULT_AGENTIC_SYSTEM_PROMPT
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = gen_cfg.get("max_tokens", 2048)
        # 生成 LLM 的 disable_thinking 仅在 vLLM 模式下下发 (与 reflection 同样的契约):
        #   - disable_thinking_extra_body=True (vLLM): 通过 chat_template_kwargs 控制
        #   - 否则 (云平台): 不下发任何 thinking 控制参数, 保持云端默认行为
        gen_extra_body = bool(gen_cfg.get("disable_thinking_extra_body", False))
        if gen_extra_body:
            disable_thinking: Optional[bool] = bool(gen_cfg.get("disable_thinking", True))
        else:
            disable_thinking = None

        lg_cfg = self.config.retrieval.get("langgraph", {}) or {}
        fallback_on_error = bool(lg_cfg.get("fallback_on_error", False))

        t0 = time.time()
        try:
            run_result = agent.run(
                query,
                history=history,
                session_meta=session.last_turn_meta() if session else {},
            )
        except Exception as e:
            if fallback_on_error:
                logger.warning(
                    f"[langgraph] agent.run 失败, 按 fallback_on_error 降级到 "
                    f"legacy agentic pipeline: {e}"
                )
                return self._run_agentic(
                    query, stream=stream, output_file=output_file,
                    history=history, session=session,
                )
            elapsed = time.time() - t0
            return QueryResult(query=query, error=str(e), latency_s=round(elapsed, 3))

        # FC clarify 出口: 直接返回反问, 跳过检索后的 LLM 生成
        if run_result.get("needs_clarify"):
            clarify_answer = str(
                run_result.get("answer") or run_result.get("context") or ""
            ).strip()
            elapsed = time.time() - t0
            logger.info(
                f"[langgraph-clarify] 跳过生成, 直接返回反问: {clarify_answer[:80]!r}"
            )
            # 关键: 本轮没做检索, this_round_docs 为空; 直接写 session_meta={}
            # 会把上一轮的 doc_registry 也覆盖掉, 导致用户回答澄清问题后 router
            # 失去 "第X篇" 锚点. 所以这里把上一轮的 session_meta 原样透传下去.
            carry_meta = session.last_turn_meta() if session else {}
            clarify_meta: Dict[str, Any] = dict(carry_meta or {})
            clarify_meta["clarify_pending"] = {
                "q": (run_result.get("clarify_request") or {}).get("q", ""),
                "opts": (run_result.get("clarify_request") or {}).get("opts", []),
            }
            clarify_result = QueryResult(
                query=query,
                answer=clarify_answer,
                hits=[],
                context=clarify_answer,
                usage=None,
                latency_s=round(elapsed, 3),
                session_meta=clarify_meta,
                needs_clarify=True,
                correlation_id=run_result.get("correlation_id", ""),
            )
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(clarify_result.model_dump(), f, ensure_ascii=False, indent=2)
                logger.info(f"结果已写入: {output_file}")
            return clarify_result

        if run_result.get("no_answer"):
            answer = str(run_result.get("answer") or run_result.get("context") or "").strip()
            elapsed = time.time() - t0
            logger.info(
                f"[langgraph-no-answer] 证据不足, 跳过生成: {answer[:80]!r}"
            )
            no_answer_result = QueryResult(
                query=query,
                answer=answer,
                hits=[],
                context=str(run_result.get("context", "")),
                usage=None,
                latency_s=round(elapsed, 3),
                session_meta=session.last_turn_meta() if session else {},
                no_answer=True,
                retry_count=int(run_result.get("retry_count", 0) or 0),
                correlation_id=run_result.get("correlation_id", ""),
            )
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(no_answer_result.model_dump(), f, ensure_ascii=False, indent=2)
                logger.info(f"结果已写入: {output_file}")
            return no_answer_result

        # P0-1: FC reuse 出口 — 跳过检索, 但仍调用生成 LLM, 只是 user template 不同
        is_reuse = bool(run_result.get("needs_reuse"))
        context = run_result.get("context", "")
        if is_reuse:
            user_msg = REUSE_USER_TEMPLATE.format(context=context, query=query)
            logger.info(
                f"[langgraph-reuse] mode={(run_result.get('reuse_request') or {}).get('mode')} "
                f"op={(run_result.get('reuse_request') or {}).get('op', '')[:80]!r}"
            )
        else:
            user_msg = AGENTIC_USER_TEMPLATE.format(context=context)

        logger.info(
            f"[langgraph-generate] prompt_chars={len(user_msg)} "
            f"retry_count={run_result.get('retry_count', 0)} "
            f"correlation_id={run_result.get('correlation_id', '')}"
        )

        llm = self._get_simple_llm()
        t_gen_start = time.time()
        ttft: Optional[float] = None

        try:
            if history:
                messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_msg})
                if stream:
                    answer_parts = []
                    for piece in llm.chat_messages_stream(
                        messages, temperature=temperature, max_tokens=max_tokens,
                        disable_thinking=disable_thinking,
                    ):
                        if ttft is None:
                            ttft = time.time() - t_gen_start
                            print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                        answer_parts.append(piece)
                        print(piece, end="", flush=True)
                    print()
                    answer = "".join(answer_parts)
                    usage = None
                else:
                    chat_res = llm.chat_messages(
                        messages, temperature=temperature, max_tokens=max_tokens,
                        disable_thinking=disable_thinking,
                    )
                    answer = chat_res["answer"]
                    usage = chat_res.get("usage")
                    if not str(answer or "").strip():
                        raise RuntimeError("generation LLM 返回空内容")
            else:
                if stream:
                    answer_parts = []
                    for piece in llm.chat_stream(
                        system=system_prompt, user=user_msg,
                        temperature=temperature, max_tokens=max_tokens,
                        disable_thinking=disable_thinking,
                    ):
                        if ttft is None:
                            ttft = time.time() - t_gen_start
                            print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                        answer_parts.append(piece)
                        print(piece, end="", flush=True)
                    print()
                    answer = "".join(answer_parts)
                    usage = None
                else:
                    chat_res = llm.chat(
                        system=system_prompt, user=user_msg,
                        temperature=temperature, max_tokens=max_tokens,
                        disable_thinking=disable_thinking,
                    )
                    answer = chat_res["answer"]
                    usage = chat_res.get("usage")
                    if not str(answer or "").strip():
                        raise RuntimeError("generation LLM 返回空内容")
        except Exception as e:
            logger.error(f"[langgraph-generate] LLM 失败: {e}")
            elapsed = time.time() - t0
            return QueryResult(query=query, error=str(e), latency_s=round(elapsed, 3))

        t_gen = time.time() - t_gen_start
        elapsed = time.time() - t0

        # 序列化 hits
        hits_data: List[Dict[str, Any]] = []
        for route, res in run_result.get("results", {}).items():
            if hasattr(res, "chunk_hits"):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
            elif isinstance(res, list):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])

        latency = run_result.get("latency", {})
        logger.info(
            f"[耗时-langgraph] route={latency.get('route_s', 0):.2f}s | "
            f"retrieve={latency.get('retrieve_s', 0):.2f}s | "
            f"reranker={latency.get('reranker_s', 0):.2f}s | "
            f"reflect={latency.get('reflect_s', 0):.2f}s | "
            f"rewrite={latency.get('rewrite_s', 0):.2f}s | "
            f"generate={t_gen:.2f}s | "
            f"total={elapsed:.2f}s | "
            f"retry={run_result.get('retry_count', 0)} | "
            f"reranker_score={run_result.get('reranker_score', 0):.4f}"
        )

        # 跨轮 doc_registry 持久化 (issue #1) + last_context/last_answer 用于下轮 reuse (P0-1)
        doc_registry = run_result.get("doc_registry", []) or []
        # reuse 出口: persist_last_context 是上一轮原值 (本轮没新检索), 应继续保留;
        # 普通检索出口: persist_last_context 是本轮新生成的, 本轮 answer 也写回.
        persisted_ctx = run_result.get("persist_last_context", "") or ""
        if is_reuse:
            # reuse 路径 answer 是 LLM 基于上轮材料重写; 保持 last_context 不变,
            # last_answer 则更新为本轮 answer (供下下轮判断 continue/drilldown 等).
            persisted_answer = _truncate_persist(answer, 1500)
        else:
            persisted_answer = _truncate_persist(answer, 1500)
        new_meta: Dict[str, Any] = {
            "doc_registry": doc_registry,
            "last_context": persisted_ctx,
            "last_answer": persisted_answer,
        }
        # clarify_pending 一旦用户回答, 本轮 router 已消费, 不再传递
        # (若用户当轮又触发了一次新的 clarify, 已经走上面的 clarify 分支了)

        query_result = QueryResult(
            query=query,
            answer=answer,
            hits=hits_data,
            context=context,
            usage=usage,
            latency_s=round(elapsed, 3),
            session_meta=new_meta,
            needs_reuse=is_reuse,
            no_answer=bool(run_result.get("no_answer", False)),
            retry_count=int(run_result.get("retry_count", 0) or 0),
            correlation_id=run_result.get("correlation_id", ""),
        )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_result.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")

        return query_result

    # ── 专业研究模式 (professional) ─────────────────────────────────────

    def _get_research_agent(self):
        """懒加载专业研究模式 agent (独立子图, 复用 pipeline 组件 + reranker)。"""
        if self._research_agent is not None:
            return self._research_agent

        from ..retrieval.research_agent import build_research_agent_from_pipeline
        from ..retrieval.reflect_summary import ReflectSummaryConfig
        from ..clients.reranker import RerankerClient

        pipeline = self._get_agentic_pipeline()
        ret_cfg = self.config.retrieval
        lg_cfg = ret_cfg.get("langgraph", {}) or {}
        prof_cfg = lg_cfg.get("professional", {}) or {}
        rerank_cfg = lg_cfg.get("reranker", {}) or {}

        # 规划/policy 复用生成 LLM (Qwen3.5-9B 等, 支持 FC + thinking)
        llm = self._get_simple_llm()

        # reranker: 复用普通模式的 reranker 配置 (启用时构建)
        reranker_client = None
        if rerank_cfg.get("enabled", False):
            try:
                reranker_client = RerankerClient(
                    api_base=rerank_cfg.get("api_base", "http://localhost:8001/v1"),
                    model=rerank_cfg.get("model", "Qwen/Qwen3-Reranker-0.6B"),
                    api_key=rerank_cfg.get("api_key") or self.config.generation.get("api_key", ""),
                    top_k=int(rerank_cfg.get("top_k", 5)),
                    score_threshold=float(rerank_cfg.get("quality_threshold", 0.5)),
                    timeout=int(rerank_cfg.get("timeout", 60)),
                    max_retries=int(rerank_cfg.get("max_retries", 2)),
                    fail_open=bool(rerank_cfg.get("fail_open", True)),
                )
            except Exception as e:
                logger.warning(f"[research] reranker 初始化失败, 禁用: {e}")

        summary_ret_cfg = summary_config_from_dict(ret_cfg.get("summary"))

        # rerank 门控与快速检索保持一致: 复用同一套 (route × stage × type) 阈值矩阵、
        # by_type 阈值与 fail-open 安全网, 而非只用单一 quality_threshold。
        from ..retrieval.quality_thresholds import RouteThresholds
        qt_by_type_raw = rerank_cfg.get("quality_threshold_by_type") or {}
        rr_qt_by_type: Optional[Dict[str, float]] = None
        if qt_by_type_raw:
            rr_qt_by_type = {str(k).lower(): float(v) for k, v in qt_by_type_raw.items()}
        rr_route_thresholds = RouteThresholds.from_dict(
            rerank_cfg.get("quality_thresholds"),
            legacy_default=float(rerank_cfg.get("quality_threshold", 0.25)),
            legacy_by_type=rr_qt_by_type,
        )
        rr_fail_open = rerank_cfg.get("fail_open_min_emb_quality")
        if rr_fail_open is not None:
            rr_fail_open = float(rr_fail_open)

        def _opt_bool(key: str):
            v = prof_cfg.get(key, False)
            return None if v is None else bool(v)

        # ── 文件定义式 skill 加载 (选不到 skill 时下游回退通用逻辑) ──
        from ..routing.research_skills import load_skills, resolve_skills_config
        sk = resolve_skills_config(prof_cfg.get("skills", {}) or {})
        loaded_skills: Dict[str, Any] = {}
        skill_router_mode = "off"
        skill_router_max_tokens = sk["router_max_tokens"]
        skill_router_disable_thinking = sk["router_disable_thinking"]
        skill_router_min_confidence = sk["router_min_confidence"]
        skill_router_strong_min_hits = sk["router_strong_min_hits"]
        if sk["enabled"]:
            loaded_skills = load_skills(sk["dirs"])
            skill_router_mode = sk["router_mode"]

        self._research_agent = build_research_agent_from_pipeline(
            pipeline,
            planner_llm=llm,
            policy_llm=llm,
            reranker_client=reranker_client,
            max_batches=int(prof_cfg.get("max_batches_per_round", 3)),
            max_rounds=int(prof_cfg.get("max_rounds", 4)),
            stall_limit=int(prof_cfg.get("stall_limit", 2)),
            planner_max_tokens=int(prof_cfg.get("planner_max_tokens", 2048)),
            policy_max_tokens=int(prof_cfg.get("policy_max_tokens", 2048)),
            planner_disable_thinking=_opt_bool("planner_disable_thinking"),
            policy_disable_thinking=_opt_bool("policy_disable_thinking"),
            reranker_top_k=int(rerank_cfg.get("top_k", 5)),
            reranker_quality_k=int(rerank_cfg.get("quality_k", 3)),
            reranker_quality_threshold=float(rerank_cfg.get("quality_threshold", 0.5)),
            reranker_quality_threshold_by_type=rr_qt_by_type,
            reranker_route_thresholds=rr_route_thresholds,
            fail_open_min_emb_quality=rr_fail_open,
            summary_top_docs=summary_ret_cfg.top_docs,
            summary_per_query_k=summary_ret_cfg.per_query_k,
            synthesis_snippet_chars=int(prof_cfg.get("synthesis_snippet_chars", 500)),
            stall_quality_floor=(
                float(prof_cfg["stall_quality_floor"])
                if prof_cfg.get("stall_quality_floor") is not None else None
            ),
            obs_summary_max_chars=int(prof_cfg.get("obs_summary_max_chars", 1800)),
            gap_stall_limit=int(prof_cfg.get("gap_stall_limit", 2)),
            skills=loaded_skills or None,
            skill_router_llm=llm,
            skill_router_mode=skill_router_mode,
            skill_router_max_tokens=skill_router_max_tokens,
            skill_router_disable_thinking=skill_router_disable_thinking,
            skill_router_min_confidence=skill_router_min_confidence,
            skill_router_strong_min_hits=skill_router_strong_min_hits,
        )
        logger.info(
            f"[research] ResearchAgent 就绪: max_rounds={prof_cfg.get('max_rounds', 4)} "
            f"max_batches={prof_cfg.get('max_batches_per_round', 3)} "
            f"reranker={'on' if reranker_client else 'off'} "
            f"skills={sorted(loaded_skills) if loaded_skills else 'off'}"
        )
        return self._research_agent

    def _run_research(
        self,
        query: str,
        stream: bool = False,
        output_file: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session: Optional[ChatSession] = None,
    ) -> QueryResult:
        """专业研究模式: 多轮递进式检索闭环 + 综述式综合生成。"""
        from ..retrieval.research_agent import (
            RESEARCH_SYNTHESIS_SYSTEM,
            RESEARCH_SYNTHESIS_USER_TEMPLATE,
        )

        gen_cfg = self.config.generation
        lg_cfg = self.config.retrieval.get("langgraph", {}) or {}
        fallback_on_error = bool(lg_cfg.get("fallback_on_error", False))
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = int(gen_cfg.get("max_tokens", 2048))
        gen_extra_body = bool(gen_cfg.get("disable_thinking_extra_body", False))
        disable_thinking: Optional[bool] = (
            bool(gen_cfg.get("disable_thinking", True)) if gen_extra_body else None
        )

        t0 = time.time()
        try:
            agent = self._get_research_agent()
            run_result = agent.run(
                query,
                history=history,
                session_meta=session.last_turn_meta() if session else {},
            )
        except Exception as e:
            if fallback_on_error:
                logger.warning(
                    f"[research] 失败, 按 fallback_on_error 降级到普通 LangGraph: {e}"
                )
                return self._run_langgraph(
                    query, stream=stream, output_file=output_file,
                    history=history, session=session,
                )
            elapsed = time.time() - t0
            return QueryResult(query=query, error=str(e), latency_s=round(elapsed, 3))

        # 规划前置兜底 (无关/闲聊): 直接返回, 不检索不综述
        if run_result.get("research_status") == "reject" or run_result.get("direct_answer"):
            reject_answer = str(
                run_result.get("direct_answer") or run_result.get("answer") or ""
            ).strip() or "这个问题和当前文献库主题不太相关，欢迎提出与库内文献相关的研究问题。"
            elapsed = time.time() - t0
            return QueryResult(
                query=query, answer=reject_answer, hits=[], context="",
                latency_s=round(elapsed, 3),
                session_meta=session.last_turn_meta() if session else {},
                correlation_id=run_result.get("correlation_id", ""),
                research=_research_meta(run_result),
            )

        # 研究模式触发反问: 直接返回, 不生成
        if run_result.get("needs_clarify"):
            clarify_answer = str(run_result.get("answer") or "").strip()
            elapsed = time.time() - t0
            carry_meta = dict((session.last_turn_meta() if session else {}) or {})
            carry_meta["clarify_pending"] = {
                "q": (run_result.get("clarify_request") or {}).get("q", ""),
                "opts": (run_result.get("clarify_request") or {}).get("opts", []),
            }
            # 研究模式 clarify 发生在检索之后: 已找到的文献随 registry 持久化, 便于回指
            found_docs = run_result.get("doc_registry") or []
            if found_docs:
                carry_meta["doc_registry"] = found_docs
            return QueryResult(
                query=query, answer=clarify_answer, hits=[],
                context=clarify_answer, latency_s=round(elapsed, 3),
                session_meta=carry_meta, needs_clarify=True,
                correlation_id=run_result.get("correlation_id", ""),
                research=_research_meta(run_result),
            )

        context = run_result.get("context", "") or ""

        # 无任何证据: 保守说明, 不生成 (避免编造)
        if run_result.get("no_answer") or not context.strip():
            elapsed = time.time() - t0
            answer = (
                "我在当前文献库中没有检索到足以支撑这个研究问题的证据，"
                "因此不便给出综述结论。你可以补充更具体的研究方向、材料/工艺名或关键文献后再试。"
            )
            return QueryResult(
                query=query, answer=answer, hits=[], context=context,
                latency_s=round(elapsed, 3),
                session_meta=session.last_turn_meta() if session else {},
                no_answer=True,
                correlation_id=run_result.get("correlation_id", ""),
                research=_research_meta(run_result),
            )

        user_msg = RESEARCH_SYNTHESIS_USER_TEMPLATE.format(context=context)
        llm = self._get_simple_llm()
        t_gen_start = time.time()

        try:
            if stream:
                answer_parts: List[str] = []
                for piece in llm.chat_stream(
                    system=RESEARCH_SYNTHESIS_SYSTEM, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    answer_parts.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(answer_parts)
                usage = None
            else:
                chat_res = llm.chat(
                    system=RESEARCH_SYNTHESIS_SYSTEM, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
                answer = chat_res["answer"]
                usage = chat_res.get("usage")
                if not str(answer or "").strip():
                    raise RuntimeError("research synthesis LLM 返回空内容")
        except Exception as e:
            logger.error(f"[research-generate] LLM 失败: {e}")
            elapsed = time.time() - t0
            return QueryResult(query=query, error=str(e), latency_s=round(elapsed, 3))

        t_gen = time.time() - t_gen_start
        elapsed = time.time() - t0

        hits_data = _research_hits(run_result)

        lat = run_result.get("latency", {})
        logger.info(
            f"[耗时-research] plan={lat.get('plan_s', 0):.2f}s | "
            f"retrieve={lat.get('retrieve_s', 0):.2f}s | "
            f"reranker={lat.get('reranker_s', 0):.2f}s | "
            f"policy={lat.get('research_policy_s', 0):.2f}s | "
            f"generate={t_gen:.2f}s | total={elapsed:.2f}s | "
            f"rounds={run_result.get('research_rounds', 0)} | "
            f"evidence_docs={run_result.get('evidence_doc_count', 0)}"
        )

        new_meta: Dict[str, Any] = {
            "doc_registry": run_result.get("doc_registry", []) or [],
            "last_context": run_result.get("persist_last_context", "") or "",
            "last_answer": _truncate_persist(answer, 1500),
            "research_carryover": run_result.get("research_carryover") or {},  # #6/#8/#10
        }
        query_result = QueryResult(
            query=query,
            answer=answer,
            hits=hits_data,
            context=context,
            usage=usage,
            latency_s=round(elapsed, 3),
            session_meta=new_meta,
            correlation_id=run_result.get("correlation_id", ""),
            research=_research_meta(run_result),
        )
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_result.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")
        return query_result

    def _run_agentic(
        self,
        query: str,
        stream: bool = False,
        output_file: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session: Optional[ChatSession] = None,
    ) -> QueryResult:
        """Agentic RAG 模式: LLM 路由 + 多路径并行检索 + 生成。"""
        gen_cfg = self.config.generation

        pipeline = self._get_agentic_pipeline()

        system_prompt = gen_cfg.get("system_prompt") or DEFAULT_AGENTIC_SYSTEM_PROMPT
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = gen_cfg.get("max_tokens", 2048)
        disable_thinking = bool(gen_cfg.get("disable_thinking", True))

        t0 = time.time()
        try:
            result = pipeline.answer(
                query, system=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
                stream=stream, history=history,
                chat_messages=history,
            )
        except Exception as e:
            elapsed = time.time() - t0
            return QueryResult(query=query, error=str(e), latency_s=round(elapsed, 3))

        elapsed = time.time() - t0
        answer = result.get("answer", "")
        usage = result.get("usage")

        # 序列化 hits
        hits_data: List[Dict[str, Any]] = []
        for route, res in result.get("results", {}).items():
            if hasattr(res, "chunk_hits"):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
            elif isinstance(res, list):
                hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])

        query_result = QueryResult(
            query=query,
            answer=answer,
            hits=hits_data,
            context=result.get("context", ""),
            usage=usage,
            latency_s=round(elapsed, 3),
        )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_result.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")

        return query_result

    def _run_simple(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        output_file: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session: Optional[ChatSession] = None,
    ) -> QueryResult:
        """简单模式: 检索 + 基础生成 (非 Agentic)。"""
        gen_cfg = self.config.generation
        ret_cfg = self.config.retrieval

        mode = mode or ret_cfg.get("mode", "hybrid")
        top_k = top_k or ret_cfg.get("top_k", 3)
        per_retriever_k = ret_cfg.get("per_retriever_k", 10)
        doc_id = ret_cfg.get("doc_id")
        chunk_type = ret_cfg.get("chunk_type")

        t0 = time.time()

        # 检索 (复用懒加载的 retrievers)
        meta, vec, bm25, hybrid = self._get_simple_retrievers()

        retriever_map = {
            "metadata": meta, "vector": vec, "bm25": bm25, "hybrid": hybrid,
        }
        retriever = retriever_map.get(mode, hybrid)
        filter_expr = _build_filter_expr(doc_id, chunk_type)

        retrieve_kwargs: Dict[str, Any] = {
            "top_k": top_k,
            "filter_expr": filter_expr,
        }
        if mode == "hybrid":
            retrieve_kwargs["per_retriever_k"] = per_retriever_k
            hybrid_cfg = hybrid_config_from_dict(ret_cfg.get("hybrid"))
            chunk_type_val = ret_cfg.get("chunk_type")
            bias = infer_retrieve_bias_heuristic(query, chunk_type=chunk_type_val)
            weights = infer_hybrid_weights(
                STAGE_SIMPLE, query, retrieve_bias=bias, chunk_type=chunk_type_val,
                config=hybrid_cfg,
            )
            retrieve_kwargs["dense_weight"] = weights.dense
            retrieve_kwargs["bm25_weight"] = weights.bm25
            logger.info(f"[hybrid] {format_weight_log(weights)}")

        t_ret_start = time.time()
        hits = retriever.retrieve(query, **retrieve_kwargs)
        t_retrieve = time.time() - t_ret_start

        t_render_start = time.time()
        context = ContextBuilder().build(hits, query=query)
        t_render = time.time() - t_render_start
        logger.info(f"[{mode}] retrieved {len(hits)} hits")
        logger.info(f"[检索上下文] (长度 {len(context)} 字符)\n{context}")

        # 生成 (复用懒加载的 LLM client)
        llm = self._get_simple_llm()

        system_prompt = gen_cfg.get("system_prompt", "")
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = gen_cfg.get("max_tokens", 2048)
        disable_thinking = bool(gen_cfg.get("disable_thinking", True))

        user_msg = _build_user_message(context, query)
        t_gen_start = time.time()
        ttft: Optional[float] = None

        if history:
            # 多轮: 构建完整 messages 列表
            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_msg})
            if stream:
                answer_parts = []
                for piece in llm.chat_messages_stream(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    if ttft is None:
                        ttft = time.time() - t_gen_start
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    answer_parts.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(answer_parts)
                usage = None
            else:
                res = llm.chat_messages(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
                answer = res["answer"]
                usage = res.get("usage")
        else:
            # 单轮
            if stream:
                answer_parts = []
                for piece in llm.chat_stream(
                    system=system_prompt, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    if ttft is None:
                        ttft = time.time() - t_gen_start
                        print(f"\n[首包] ttft={ttft:.2f}s", flush=True)
                    answer_parts.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(answer_parts)
                usage = None
            else:
                res = llm.chat(
                    system=system_prompt, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
                answer = res["answer"]
                usage = res.get("usage")
        t_gen = time.time() - t_gen_start
        elapsed = time.time() - t0

        gen_part = (
            f"generate={t_gen:.2f}s"
            + (f" (ttft={ttft:.2f}s)" if ttft is not None else "")
        )
        logger.info(
            f"[耗时-端到端 simple] retrieve={t_retrieve:.2f}s | "
            f"render={t_render:.2f}s | {gen_part} | "
            f"total={elapsed:.2f}s"
        )

        hits_data = [asdict(h) for h in hits]

        query_result = QueryResult(
            query=query,
            answer=answer,
            hits=hits_data,
            context=context,
            usage=usage,
            latency_s=round(elapsed, 3),
        )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_result.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"结果已写入: {output_file}")

        return query_result

    def _stream_research_events(
        self,
        query: str,
        *,
        session: ChatSession,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """专业研究模式的 SSE 事件: 多轮检索闭环 + 流式综述生成。"""
        from ..retrieval.research_agent import (
            RESEARCH_SYNTHESIS_SYSTEM,
            RESEARCH_SYNTHESIS_USER_TEMPLATE,
            RESEARCH_THINKING_SYSTEM,
            RESEARCH_THINKING_USER_TEMPLATE,
        )

        gen_cfg = self.config.generation
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = int(gen_cfg.get("max_tokens", 2048))

        prof_cfg = (
            (self.config.retrieval.get("langgraph", {}) or {}).get("professional", {}) or {}
        )
        # 综述阶段开思考: 流式展示"思考过程"再输出正文; 开思考后思考+正文都吃 token,
        # 故用更大的 synthesis_max_tokens 预算, 避免正文被截断。
        show_thinking = bool(prof_cfg.get("synthesis_show_thinking", True))
        synth_max_tokens = int(prof_cfg.get("synthesis_max_tokens", max(max_tokens, 6000)))

        t0 = time.time()
        run_result: Optional[Dict[str, Any]] = None
        try:
            for kind, payload in self._get_research_agent().run_events(
                query, history=history,
                session_meta=session.last_turn_meta(),
            ):
                if kind == "thinking":
                    # 实时"思考过程": 规划如何拆解 → 每轮评估/缺口 → 为何继续或收口
                    yield {"type": "thinking", **payload}
                elif kind == "result":
                    run_result = payload
        except Exception as e:
            logger.exception("[research-stream] 检索失败")
            yield {"type": "error", "message": str(e)}
            return

        if run_result is None:
            yield {"type": "error", "message": "研究流程未产出结果"}
            return

        # 规划前置兜底 (无关/闲聊/无意义): 直接回复, 不检索不综述, 也不挂 clarify_pending
        if run_result.get("research_status") == "reject" or run_result.get("direct_answer"):
            reject_answer = str(
                run_result.get("direct_answer") or run_result.get("answer") or ""
            ).strip() or "这个问题和当前文献库主题不太相关，欢迎提出与库内文献相关的研究问题。"
            elapsed = time.time() - t0
            yield {"type": "text", "content": reject_answer}
            yield {
                "type": "done", "answer": reject_answer, "hits": [],
                "context": "", "usage": None,
                "latency_s": round(elapsed, 3),
                "session_meta": session.last_turn_meta() or {},
                "needs_clarify": False, "needs_reuse": False, "retry_count": 0,
                "correlation_id": run_result.get("correlation_id", ""),
                "research": _research_meta(run_result),
            }
            return

        if run_result.get("needs_clarify"):
            clarify_answer = str(run_result.get("answer") or "").strip()
            carry_meta = dict(session.last_turn_meta() or {})
            carry_meta["clarify_pending"] = {
                "q": (run_result.get("clarify_request") or {}).get("q", ""),
                "opts": (run_result.get("clarify_request") or {}).get("opts", []),
            }
            found_docs = run_result.get("doc_registry") or []
            if found_docs:
                carry_meta["doc_registry"] = found_docs
            elapsed = time.time() - t0
            yield {"type": "text", "content": clarify_answer}
            yield {
                "type": "done", "answer": clarify_answer, "hits": [],
                "context": clarify_answer, "usage": None,
                "latency_s": round(elapsed, 3), "session_meta": carry_meta,
                "needs_clarify": True, "needs_reuse": False, "retry_count": 0,
                "correlation_id": run_result.get("correlation_id", ""),
                "research": _research_meta(run_result),
            }
            return

        context = run_result.get("context", "") or ""
        if run_result.get("no_answer") or not context.strip():
            answer = (
                "我在当前文献库中没有检索到足以支撑这个研究问题的证据，"
                "因此不便给出综述结论。你可以补充更具体的研究方向或关键文献后再试。"
            )
            elapsed = time.time() - t0
            yield {"type": "text", "content": answer}
            yield {
                "type": "done", "answer": answer, "hits": [], "context": context,
                "usage": None, "latency_s": round(elapsed, 3),
                "session_meta": session.last_turn_meta() or {},
                "needs_clarify": False, "needs_reuse": False, "retry_count": 0,
                "correlation_id": run_result.get("correlation_id", ""),
                "no_answer": True,
                "research": _research_meta(run_result),
            }
            return

        hits_data = _research_hits(run_result)

        # skill 专属综述模板 (缺失字段回退通用模板)
        skill_synth = run_result.get("skill_synthesis") or {}
        syn_system = skill_synth.get("system") or RESEARCH_SYNTHESIS_SYSTEM
        syn_user_tpl = skill_synth.get("user_template") or RESEARCH_SYNTHESIS_USER_TEMPLATE
        syn_think_system = skill_synth.get("thinking_system") or RESEARCH_THINKING_SYSTEM

        yield {"type": "status", "stage": "generating"}
        user_msg = syn_user_tpl.format(context=context)
        llm = self._get_simple_llm()
        answer_parts: List[str] = []
        try:
            if show_thinking:
                # 思考过程: 本地模型原生 reasoning 恒为英文且无法用 prompt 纠正, 故改由模型
                # 显式产出一段"中文分析思路"作为可读思考过程 (关思考直出, 内容即中文思考)。
                think_user = RESEARCH_THINKING_USER_TEMPLATE.format(context=context)
                think_max = int(prof_cfg.get("synthesis_thinking_max_tokens", 700))
                try:
                    for piece in llm.chat_stream(
                        system=syn_think_system, user=think_user,
                        temperature=temperature, max_tokens=think_max,
                        disable_thinking=True,
                    ):
                        yield {"type": "thinking", "content": piece, "phase": "synthesis"}
                except Exception as e:
                    logger.warning(f"[research-stream] 中文分析思路生成失败, 跳过: {e}")
                # 正文: 关思考直出 (既避免暴露英文 CoT, 又省去英文思考占用的时延/预算)
                for piece in llm.chat_stream(
                    system=syn_system, user=user_msg,
                    temperature=temperature, max_tokens=synth_max_tokens,
                    disable_thinking=True,
                ):
                    answer_parts.append(piece)
                    yield {"type": "text", "content": piece}
            else:
                for piece in llm.chat_stream(
                    system=syn_system, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=True,
                ):
                    answer_parts.append(piece)
                    yield {"type": "text", "content": piece}
        except Exception as e:
            logger.error(f"[research-stream] 生成失败: {e}")
            yield {"type": "error", "message": str(e)}
            return

        answer = "".join(answer_parts)
        elapsed = time.time() - t0
        new_meta: Dict[str, Any] = {
            "doc_registry": run_result.get("doc_registry", []) or [],
            "last_context": run_result.get("persist_last_context", "") or "",
            "last_answer": _truncate_persist(answer, 1500),
            "research_carryover": run_result.get("research_carryover") or {},  # #6/#8/#10
        }
        yield {
            "type": "done", "answer": answer, "hits": hits_data, "context": context,
            "usage": None, "latency_s": round(elapsed, 3), "session_meta": new_meta,
            "needs_clarify": False, "needs_reuse": False, "retry_count": 0,
            "correlation_id": run_result.get("correlation_id", ""),
            "research": _research_meta(run_result),
        }

    # ── SSE 流式对话 ────────────────────────────────────────────────────

    def stream_chat_events(
        self,
        query: str,
        session: ChatSession,
        use_agentic: bool = True,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        professional: bool = False,
        collection: Optional[str] = None,
    ):
        """SSE 流式对话: 检索完成后逐块 yield LLM 输出。

        Yields:
            dict — 每个事件:
              {"type": "status", "stage": "retrieving" | "generating"}
              {"type": "text",  "content": "..."}
              {"type": "done",  "hits": [...], "usage": ..., "latency_s": ...,
                               "session_meta": {...}, "answer": "完整回复"}
              {"type": "error", "message": "..."}
        """
        # 注: 目标集合切换 (含 collection=None 回退默认库) 已由上层
        # Pipeline._maybe_switch_collection 统一处理, 这里不再重复切换,
        # 以免遗漏"空集合回退原始默认库"的逻辑导致沿用被污染的集合。
        _ = collection

        gen_cfg = self.config.generation
        history = session.recent_messages() or None
        lg_cfg = self.config.retrieval.get("langgraph", {}) or {}
        use_langgraph = use_agentic and bool(lg_cfg.get("enabled", False))

        fallback_on_error = bool(lg_cfg.get("fallback_on_error", False))

        prof_cfg = lg_cfg.get("professional", {}) or {}
        use_professional = (
            professional and use_langgraph and bool(prof_cfg.get("enabled", True))
        )

        yield {"type": "status", "stage": "retrieving"}

        # ── 专业研究模式: 独立闭环, 不走下方普通流式路径 ──
        if use_professional:
            yield from self._stream_research_events(
                query, session=session, history=history,
            )
            return

        t0 = time.time()
        try:
            if use_langgraph:
                try:
                    run_result = self._get_langgraph_agent().run(
                        query,
                        history=history,
                        session_meta=session.last_turn_meta(),
                    )
                except Exception as e:
                    if not fallback_on_error:
                        raise
                    logger.warning(
                        f"[stream_chat] LangGraph 失败, 按 fallback_on_error "
                        f"降级到 legacy agentic: {e}"
                    )
                    use_langgraph = False

            if use_langgraph:
                pass  # run_result 已就绪, 继续走下方流式生成
            elif use_agentic:
                agentic = self._get_agentic_pipeline()
                run_result = agentic.answer(
                    query,
                    system=gen_cfg.get("system_prompt") or DEFAULT_AGENTIC_SYSTEM_PROMPT,
                    temperature=gen_cfg.get("temperature", 0),
                    max_tokens=gen_cfg.get("max_tokens", 2048),
                    stream=False,
                    history=history,
                    chat_messages=history,
                )
                # agentic pipeline 已经生成了 answer, 直接返回
                answer = run_result.get("answer", "")
                hits_data: List[Dict[str, Any]] = []
                for _route, res in run_result.get("results", {}).items():
                    if hasattr(res, "chunk_hits"):
                        hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
                    elif isinstance(res, list):
                        hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])
                elapsed = time.time() - t0
                yield {
                    "type": "done",
                    "answer": answer,
                    "hits": hits_data,
                    "context": run_result.get("context", ""),
                    "usage": run_result.get("usage"),
                    "latency_s": round(elapsed, 3),
                    "session_meta": {},
                    "needs_clarify": False,
                    "needs_reuse": False,
                    "retry_count": int(run_result.get("retry_count", 0) or 0),
                    "correlation_id": run_result.get("correlation_id", ""),
                }
                return
            else:
                # simple 模式: 不走 agent, 直接检索 + 生成
                run_result = self._simple_retrieve(query, mode=mode, top_k=top_k)
                context = run_result["context"]
                hits_data = run_result["hits"]
        except Exception as e:
            logger.exception("[stream_chat] 检索失败")
            yield {"type": "error", "message": str(e)}
            return

        # ── LangGraph 路径: 拿到 context 后流式生成 ────────────────────
        if use_langgraph:
            # clarify 出口: 不走 LLM 生成
            if run_result.get("needs_clarify"):
                clarify_answer = str(
                    run_result.get("answer") or run_result.get("context") or ""
                ).strip()
                carry_meta = session.last_turn_meta() or {}
                clarify_meta = dict(carry_meta)
                clarify_meta["clarify_pending"] = {
                    "q": (run_result.get("clarify_request") or {}).get("q", ""),
                    "opts": (run_result.get("clarify_request") or {}).get("opts", []),
                }
                elapsed = time.time() - t0
                yield {"type": "text", "content": clarify_answer}
                yield {
                    "type": "done",
                    "answer": clarify_answer,
                    "hits": [],
                    "context": clarify_answer,
                    "usage": None,
                    "latency_s": round(elapsed, 3),
                    "session_meta": clarify_meta,
                    "needs_clarify": True,
                    "needs_reuse": False,
                    "retry_count": 0,
                    "correlation_id": run_result.get("correlation_id", ""),
                }
                return

            is_reuse = bool(run_result.get("needs_reuse"))
            context = run_result.get("context", "")
            if is_reuse:
                user_msg = REUSE_USER_TEMPLATE.format(context=context, query=query)
            else:
                user_msg = AGENTIC_USER_TEMPLATE.format(context=context)

            # 序列化 hits
            hits_data = []
            for _route, res in run_result.get("results", {}).items():
                if hasattr(res, "chunk_hits"):
                    hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
                elif isinstance(res, list):
                    hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])
        else:
            # simple 路径
            context = run_result["context"]
            hits_data = run_result["hits"]
            user_msg = _build_user_message(context, query)

        # ── 流式 LLM 生成 ──────────────────────────────────────────────
        yield {"type": "status", "stage": "generating"}

        system_prompt = gen_cfg.get("system_prompt") or DEFAULT_AGENTIC_SYSTEM_PROMPT
        temperature = gen_cfg.get("temperature", 0)
        max_tokens = gen_cfg.get("max_tokens", 2048)
        gen_extra_body = bool(gen_cfg.get("disable_thinking_extra_body", False))
        if gen_extra_body:
            disable_thinking: Optional[bool] = bool(gen_cfg.get("disable_thinking", True))
        else:
            disable_thinking = None

        llm = self._get_simple_llm()
        answer_parts: List[str] = []

        try:
            if history:
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_msg})
                for piece in llm.chat_messages_stream(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    answer_parts.append(piece)
                    yield {"type": "text", "content": piece}
            else:
                for piece in llm.chat_stream(
                    system=system_prompt, user=user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                ):
                    answer_parts.append(piece)
                    yield {"type": "text", "content": piece}
        except Exception as e:
            logger.error(f"[stream_chat] LLM 流式生成失败: {e}")
            yield {"type": "error", "message": str(e)}
            return

        answer = "".join(answer_parts)
        elapsed = time.time() - t0

        # 构建 session_meta (与 _run_langgraph 一致)
        if use_langgraph:
            doc_registry = run_result.get("doc_registry", []) or []
            persisted_ctx = run_result.get("persist_last_context", "") or ""
            session_meta: Dict[str, Any] = {
                "doc_registry": doc_registry,
                "last_context": persisted_ctx,
                "last_answer": _truncate_persist(answer, 1500),
            }
            is_reuse = bool(run_result.get("needs_reuse"))
            retry_count = int(run_result.get("retry_count", 0) or 0)
            correlation_id = run_result.get("correlation_id", "")
        else:
            session_meta = {}
            is_reuse = False
            retry_count = 0
            correlation_id = ""

        yield {
            "type": "done",
            "answer": answer,
            "hits": hits_data,
            "context": context,
            "usage": None,
            "latency_s": round(elapsed, 3),
            "session_meta": session_meta,
            "needs_clarify": False,
            "needs_reuse": is_reuse,
            "retry_count": retry_count,
            "correlation_id": correlation_id,
        }

    def _simple_retrieve(
        self, query: str, mode: Optional[str] = None, top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simple 模式的检索阶段 (不含 LLM 生成), 供 stream_chat_events 复用。"""
        ret_cfg = self.config.retrieval
        mode = mode or ret_cfg.get("mode", "hybrid")
        top_k = top_k or ret_cfg.get("top_k", 3)
        per_retriever_k = ret_cfg.get("per_retriever_k", 10)
        doc_id = ret_cfg.get("doc_id")
        chunk_type = ret_cfg.get("chunk_type")

        meta, vec, bm25, hybrid = self._get_simple_retrievers()
        retriever_map = {
            "metadata": meta, "vector": vec, "bm25": bm25, "hybrid": hybrid,
        }
        retriever = retriever_map.get(mode, hybrid)
        filter_expr = _build_filter_expr(doc_id, chunk_type)

        retrieve_kwargs: Dict[str, Any] = {
            "top_k": top_k,
            "filter_expr": filter_expr,
        }
        if mode == "hybrid":
            retrieve_kwargs["per_retriever_k"] = per_retriever_k

        hits = retriever.retrieve(query, **retrieve_kwargs)
        context = ContextBuilder().build(hits, query=query)
        hits_data = [asdict(h) for h in hits]
        return {"context": context, "hits": hits_data}


def _describe_thinking(v: Optional[bool]) -> str:
    """把 disable_thinking 标志转成易读日志字符串。"""
    if v is None:
        return "default(后端缺省)"
    return "off(关闭思考)" if v else "on(开启思考)"


def _research_meta(run_result: Dict[str, Any]) -> Dict[str, Any]:
    """从 ResearchAgent.run() 结果抽出给前端展示的研究概要 (精简, 不含全文)。"""
    return {
        "status": str(run_result.get("research_status") or "complete"),
        "rounds": int(run_result.get("research_rounds", 0) or 0),
        "evidence_docs": int(run_result.get("evidence_doc_count", 0) or 0),
        "evidence_chunks": int(run_result.get("evidence_chunk_count", 0) or 0),
        "gaps": list(run_result.get("research_gaps", []) or [])[:8],
        "covered": list(run_result.get("research_covered", []) or [])[:12],
    }


def _research_hits(run_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """最终 hits 取跨轮累计去重证据 (evidence_hits); 回退到最后一轮 route_results。"""
    evidence = run_result.get("evidence_hits")
    if isinstance(evidence, list) and evidence:
        return [asdict(h) if isinstance(h, Hit) else h for h in evidence]
    hits_data: List[Dict[str, Any]] = []
    for _route, res in (run_result.get("results", {}) or {}).items():
        if hasattr(res, "chunk_hits"):
            hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res.chunk_hits])
        elif isinstance(res, list):
            hits_data.extend([asdict(h) if isinstance(h, Hit) else h for h in res])
    return hits_data


def _truncate_persist(text: str, limit: int) -> str:
    """把 last_answer 等字段持久化前截断, 防止 session_meta 膨胀。"""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "...(truncated)"


def _build_user_message(context: str, query: str) -> str:
    return (
        "# 检索到的上下文\n"
        f"{context}\n\n"
        "# 用户问题\n"
        f"{query}\n\n"
        "请基于上下文给出严谨、有引用的回答。"
    )
