# Chunker Low-Confidence LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add observable chunk quality diagnostics and optional low-confidence LLM-assisted repair without replacing the existing deterministic chunker.

**Architecture:** Keep the current MinerU/UniParser chunk construction as the primary path. Add small, focused post-processing modules for diagnostics, asset linking, section repair, and optional LLM boundary splitting; each module is config-gated and returns metadata or boundary decisions rather than rewriting source text.

**Tech Stack:** Python, existing `EmbeddingClient`, existing `LLMClient`, pytest/unittest tests, YAML config in `default_config.yaml`.

---

## File Structure

- Modify `default_config.yaml`
  - Add config blocks for `quality_gate`, `asset_linking`, `section_repair`, and `llm_boundary`.
- Create `processors/chunk_diagnostics.py`
  - Computes chunk quality flags, confidence scores, and split strategy metadata.
- Create `processors/asset_linker.py`
  - Extracts figure/table references from text chunks and links them to image/table chunks by caption/number.
- Create `processors/llm_boundary_splitter.py`
  - Optional low-confidence long-text splitter that asks LLM for character/sentence boundaries only.
- Create `processors/section_repair.py`
  - Optional title/section boundary classifier for ambiguous parser output.
- Modify `processors/chunker.py`
  - Add `split_strategy` during chunk creation and call diagnostics/linking post-processors near the end of `build_knowledge_blocks`.
- Modify `processors/uniparser_chunker.py`
  - Mirror diagnostics/linking metadata for UniParser chunks.
- Modify `steps/chunk.py`
  - Parse new config sections and pass them into chunker entry points.
- Create `tests/test_chunk_diagnostics.py`
  - Unit tests for quality flags and confidence scoring.
- Create `tests/test_asset_linker.py`
  - Unit tests for figure/table reference matching.
- Create `tests/test_llm_boundary_splitter.py`
  - Unit tests using a fake LLM client; no network calls.
- Create `tests/test_chunker_diagnostics_integration.py`
  - Integration tests proving metadata is added without changing default chunk content.

---

## Phase 1: Observability First, No Behavior Change

### Task 1: Add Chunk Diagnostics Config

**Files:**
- Modify: `default_config.yaml:159`

- [ ] **Step 1: Add config defaults under `chunking`**

Add this block after `semantic_split` and before `references_batch_size`:

```yaml
  quality_gate:
    enabled: true
    min_confidence_for_accept: 0.65
    llm_review_enabled: false
    max_noise_ratio: 0.35
    min_text_chars: 30
    max_repeated_line_ratio: 0.4

  asset_linking:
    enabled: true
    llm_fallback_enabled: false
    max_link_distance_pages: 6

  section_repair:
    enabled: false
    llm_fallback_enabled: false
    ambiguous_min_confidence: 0.4
    ambiguous_max_confidence: 0.7

  llm_boundary:
    enabled: false
    only_low_confidence: true
    max_input_chars: 6000
    min_chunk_chars: 300
    max_chunk_chars: 2000
    return_boundaries_only: true
```

- [ ] **Step 2: Verify YAML parses**

Run:

```bash
python - <<'PY'
import yaml
with open('default_config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
assert cfg['chunking']['quality_gate']['enabled'] is True
assert cfg['chunking']['llm_boundary']['enabled'] is False
print('ok')
PY
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add default_config.yaml
git commit -m "config: add chunk quality controls"
```

### Task 2: Create Chunk Diagnostics Module

**Files:**
- Create: `processors/chunk_diagnostics.py`
- Test: `tests/test_chunk_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunk_diagnostics.py`:

```python
import unittest

from pipeline.processors.chunk_diagnostics import diagnose_chunk, apply_chunk_diagnostics


class TestChunkDiagnostics(unittest.TestCase):
    def test_short_text_chunk_gets_low_confidence_flag(self):
        chunk = {"id": "text_1", "type": "text", "content": "太短", "pages": [0]}

        result = diagnose_chunk(chunk, min_text_chars=30)

        self.assertIn("too_short", result["quality_flags"])
        self.assertLess(result["chunk_confidence"], 0.65)

    def test_normal_text_chunk_gets_high_confidence(self):
        chunk = {
            "id": "text_1",
            "type": "text",
            "content": "本文研究耐候钢在海洋大气环境中的腐蚀行为，并分析不同合金元素对锈层稳定性的影响。",
            "pages": [0],
        }

        result = diagnose_chunk(chunk, min_text_chars=30)

        self.assertEqual(result["quality_flags"], [])
        self.assertGreaterEqual(result["chunk_confidence"], 0.9)

    def test_metadata_added_without_changing_content(self):
        chunks = [{"id": "text_1", "type": "text", "content": "正常正文内容足够长，可以作为一个稳定的检索块。", "pages": [0]}]

        out = apply_chunk_diagnostics(chunks, enabled=True, default_strategy="paragraph_rule")

        self.assertEqual(out[0]["content"], chunks[0]["content"])
        self.assertEqual(out[0]["split_strategy"], "paragraph_rule")
        self.assertIn("chunk_confidence", out[0])
        self.assertIn("quality_flags", out[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_chunk_diagnostics.py -v
```

Expected: FAIL because `pipeline.processors.chunk_diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostics module**

Create `processors/chunk_diagnostics.py`:

```python
"""Chunk quality diagnostics and metadata enrichment.

This module is intentionally deterministic. It does not modify chunk content;
it only adds quality metadata for downstream routing, auditing, and optional
low-confidence LLM fallback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_NOISE_RE = re.compile(r"[^\w\s\u3400-\u9fff，。！？；：、,.!?;:()（）\-\[\]【】/%℃°+=<>]")
_REPEATED_LINE_MIN = 3


def _noise_ratio(text: str) -> float:
    if not text:
        return 1.0
    return len(_NOISE_RE.findall(text)) / max(1, len(text))


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < _REPEATED_LINE_MIN:
        return 0.0
    counts: Dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(1, len(lines))


def diagnose_chunk(
    chunk: Dict[str, Any],
    *,
    min_text_chars: int = 30,
    max_noise_ratio: float = 0.35,
    max_repeated_line_ratio: float = 0.4,
) -> Dict[str, Any]:
    """Return quality metadata for one chunk without mutating it."""
    chunk_type = chunk.get("type") or ""
    content = (chunk.get("content") or "").strip()
    flags: List[str] = []

    if chunk_type in {"text", "summary"} and len(content) < min_text_chars:
        flags.append("too_short")

    noise = _noise_ratio(content)
    if noise > max_noise_ratio:
        flags.append("high_noise_ratio")

    repeated = _repeated_line_ratio(content)
    if repeated > max_repeated_line_ratio:
        flags.append("repeated_lines")

    if chunk_type == "text" and not content:
        flags.append("empty_text")

    confidence = 1.0
    confidence -= 0.25 * len(flags)
    if "high_noise_ratio" in flags:
        confidence -= min(0.25, noise / 2)
    if "repeated_lines" in flags:
        confidence -= min(0.2, repeated / 2)
    confidence = max(0.0, min(1.0, round(confidence, 4)))

    return {
        "quality_flags": flags,
        "chunk_confidence": confidence,
        "quality_metrics": {
            "char_count": len(content),
            "noise_ratio": round(noise, 4),
            "repeated_line_ratio": round(repeated, 4),
        },
    }


def apply_chunk_diagnostics(
    chunks: List[Dict[str, Any]],
    *,
    enabled: bool = True,
    default_strategy: str = "paragraph_rule",
    min_text_chars: int = 30,
    max_noise_ratio: float = 0.35,
    max_repeated_line_ratio: float = 0.4,
) -> List[Dict[str, Any]]:
    """Add diagnostic metadata to chunks in place and return the same list."""
    if not enabled:
        return chunks

    for chunk in chunks:
        chunk.setdefault("split_strategy", default_strategy)
        metadata = diagnose_chunk(
            chunk,
            min_text_chars=min_text_chars,
            max_noise_ratio=max_noise_ratio,
            max_repeated_line_ratio=max_repeated_line_ratio,
        )
        chunk.update(metadata)
    return chunks
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_chunk_diagnostics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add processors/chunk_diagnostics.py tests/test_chunk_diagnostics.py
git commit -m "feat: add chunk diagnostics"
```

### Task 3: Attach Diagnostics to MinerU Chunker

**Files:**
- Modify: `processors/chunker.py:1275`
- Modify: `processors/chunker.py:1775`
- Test: `tests/test_chunker_diagnostics_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_chunker_diagnostics_integration.py`:

```python
import unittest

from pipeline.processors.chunker import build_knowledge_blocks


class TestChunkerDiagnosticsIntegration(unittest.TestCase):
    def test_build_knowledge_blocks_adds_diagnostics_metadata(self):
        data = [[
            {
                "type": "paragraph",
                "text": "本文研究耐候钢在海洋大气环境中的腐蚀行为，并分析锈层演化规律。",
                "page_idx": 0,
            }
        ]]

        chunks = build_knowledge_blocks(
            data,
            summary_enabled=False,
            doc_title="测试文献",
        )

        text_chunks = [chunk for chunk in chunks if chunk.get("type") == "text"]
        self.assertEqual(len(text_chunks), 1)
        self.assertIn("split_strategy", text_chunks[0])
        self.assertIn("chunk_confidence", text_chunks[0])
        self.assertIn("quality_flags", text_chunks[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_chunker_diagnostics_integration.py -v
```

Expected: FAIL because diagnostics metadata is not added yet.

- [ ] **Step 3: Import diagnostics and add parameters**

In `processors/chunker.py`, add import near existing imports:

```python
from .chunk_diagnostics import apply_chunk_diagnostics
```

Extend `build_knowledge_blocks` parameters:

```python
    quality_gate_enabled: bool = True,
    quality_min_text_chars: int = 30,
    quality_max_noise_ratio: float = 0.35,
    quality_max_repeated_line_ratio: float = 0.4,
```

- [ ] **Step 4: Mark semantic split strategy**

Inside `_maybe_split_text_chunk`, after `sub["chunk_total"] = len(pieces)`, add:

```python
        sub["split_strategy"] = "semantic_embedding" if embedder is not None else "semantic_greedy"
```

- [ ] **Step 5: Apply diagnostics before return**

Near the end of `build_knowledge_blocks`, after the loop that ensures `paragraph_index`, add:

```python
    apply_chunk_diagnostics(
        chunks,
        enabled=quality_gate_enabled,
        default_strategy="paragraph_rule",
        min_text_chars=quality_min_text_chars,
        max_noise_ratio=quality_max_noise_ratio,
        max_repeated_line_ratio=quality_max_repeated_line_ratio,
    )
```

- [ ] **Step 6: Run integration test**

Run:

```bash
python -m pytest tests/test_chunker_diagnostics_integration.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add processors/chunker.py tests/test_chunker_diagnostics_integration.py
git commit -m "feat: add diagnostics to mineru chunks"
```

---

## Phase 2: Deterministic Asset Linking

### Task 4: Add Figure/Table Reference Linker

**Files:**
- Create: `processors/asset_linker.py`
- Test: `tests/test_asset_linker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_asset_linker.py`:

```python
import unittest

from pipeline.processors.asset_linker import apply_asset_links, extract_asset_refs


class TestAssetLinker(unittest.TestCase):
    def test_extracts_chinese_and_english_refs(self):
        text = "如图 3 所示，腐蚀速率下降；Table 2 lists the alloy composition."

        refs = extract_asset_refs(text)

        self.assertIn(("figure", "3"), refs)
        self.assertIn(("table", "2"), refs)

    def test_links_text_chunk_to_matching_image_caption(self):
        chunks = [
            {"id": "text_1", "type": "text", "content": "如图 3 所示，锈层更加致密。", "pages": [2], "related_assets": []},
            {"id": "image_1", "type": "image", "content": "图 3 锈层形貌 SEM 图", "pages": [3], "related_assets": []},
        ]

        apply_asset_links(chunks, enabled=True, max_link_distance_pages=6)

        self.assertIn("image_1", chunks[0]["related_assets"])

    def test_does_not_link_when_page_distance_too_large(self):
        chunks = [
            {"id": "text_1", "type": "text", "content": "见表 2。", "pages": [1], "related_assets": []},
            {"id": "table_1", "type": "table", "content": "表 2 实验参数", "pages": [20], "related_assets": []},
        ]

        apply_asset_links(chunks, enabled=True, max_link_distance_pages=3)

        self.assertEqual(chunks[0]["related_assets"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_asset_linker.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement asset linker**

Create `processors/asset_linker.py`:

```python
"""Deterministic figure/table reference linking for chunks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_REF_PATTERNS = [
    ("figure", re.compile(r"(?:图|Fig\.?|Figure)\s*([0-9]+(?:[-.][0-9]+)?)", re.IGNORECASE)),
    ("table", re.compile(r"(?:表|Tab\.?|Table)\s*([0-9]+(?:[-.][0-9]+)?)", re.IGNORECASE)),
]


def extract_asset_refs(text: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    for asset_type, pattern in _REF_PATTERNS:
        for match in pattern.finditer(text or ""):
            refs.append((asset_type, match.group(1)))
    return refs


def _first_page(chunk: Dict[str, Any]) -> int:
    pages = chunk.get("pages") or []
    return int(min(pages)) if pages else -1


def _asset_kind(chunk_type: str) -> str:
    if chunk_type == "image":
        return "figure"
    if chunk_type == "table":
        return "table"
    return ""


def _build_asset_index(chunks: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for chunk in chunks:
        kind = _asset_kind(chunk.get("type") or "")
        if not kind:
            continue
        refs = extract_asset_refs(chunk.get("content") or "")
        for ref_kind, number in refs:
            if ref_kind == kind:
                index.setdefault((kind, number), []).append(chunk)
    return index


def apply_asset_links(
    chunks: List[Dict[str, Any]],
    *,
    enabled: bool = True,
    max_link_distance_pages: int = 6,
) -> List[Dict[str, Any]]:
    if not enabled:
        return chunks

    index = _build_asset_index(chunks)
    for chunk in chunks:
        if chunk.get("type") not in {"text", "summary"}:
            continue
        chunk.setdefault("related_assets", [])
        text_page = _first_page(chunk)
        for ref in extract_asset_refs(chunk.get("content") or ""):
            for asset in index.get(ref, []):
                asset_page = _first_page(asset)
                if text_page >= 0 and asset_page >= 0 and abs(text_page - asset_page) > max_link_distance_pages:
                    continue
                asset_id = asset.get("id")
                if asset_id and asset_id not in chunk["related_assets"]:
                    chunk["related_assets"].append(asset_id)
    return chunks
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_asset_linker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add processors/asset_linker.py tests/test_asset_linker.py
git commit -m "feat: add deterministic asset linking"
```

### Task 5: Attach Asset Linker to Chunkers

**Files:**
- Modify: `processors/chunker.py`
- Modify: `processors/uniparser_chunker.py`
- Modify: `steps/chunk.py`
- Test: `tests/test_asset_linker.py`

- [ ] **Step 1: Add imports**

In both chunker files, add:

```python
from .asset_linker import apply_asset_links
```

- [ ] **Step 2: Add parameters to `build_knowledge_blocks`**

Add:

```python
    asset_linking_enabled: bool = True,
    asset_linking_max_distance_pages: int = 6,
```

- [ ] **Step 3: Add parameters to `build_knowledge_blocks_uniparser`**

Add the same parameters:

```python
    asset_linking_enabled: bool = True,
    asset_linking_max_distance_pages: int = 6,
```

- [ ] **Step 4: Apply linker before diagnostics return**

Near each chunker return path, before diagnostics if diagnostics exists, call:

```python
    apply_asset_links(
        chunks,
        enabled=asset_linking_enabled,
        max_link_distance_pages=asset_linking_max_distance_pages,
    )
```

- [ ] **Step 5: Pass config from `steps/chunk.py`**

In both `_run_mineru` and `_run_uniparser`, parse:

```python
        asset_cfg = (cfg.get("asset_linking") or {}) if isinstance(cfg, dict) else {}
```

Pass into chunker calls:

```python
            asset_linking_enabled=bool(asset_cfg.get("enabled", True)),
            asset_linking_max_distance_pages=int(asset_cfg.get("max_link_distance_pages", 6)),
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
python -m pytest tests/test_asset_linker.py tests/test_chunker_diagnostics_integration.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add processors/chunker.py processors/uniparser_chunker.py steps/chunk.py
git commit -m "feat: attach asset linking to chunkers"
```

---

## Phase 3: Optional LLM Boundary Splitter

### Task 6: Create Boundary-Only LLM Splitter

**Files:**
- Create: `processors/llm_boundary_splitter.py`
- Test: `tests/test_llm_boundary_splitter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_boundary_splitter.py`:

```python
import json
import unittest

from pipeline.processors.llm_boundary_splitter import split_text_with_llm_boundaries


class FakeLLM:
    def chat(self, messages, **kwargs):
        return json.dumps({
            "boundaries": [
                {"start_char": 0, "end_char": 12, "label": "背景", "reason": "主题一"},
                {"start_char": 12, "end_char": 24, "label": "方法", "reason": "主题二"},
            ]
        }, ensure_ascii=False)


class BadLLM:
    def chat(self, messages, **kwargs):
        return "not json"


class TestLLMBoundarySplitter(unittest.TestCase):
    def test_uses_llm_boundaries_without_rewriting_text(self):
        text = "背景背景背景背景背景背景方法方法方法方法方法方法"

        pieces = split_text_with_llm_boundaries(
            text,
            FakeLLM(),
            min_chunk_chars=4,
            max_chunk_chars=20,
            max_input_chars=100,
        )

        self.assertEqual([piece.text for piece in pieces], [text[:12], text[12:24]])
        self.assertEqual(pieces[0].label, "背景")
        self.assertEqual(pieces[1].strategy, "llm_boundary")

    def test_returns_empty_on_invalid_llm_response(self):
        pieces = split_text_with_llm_boundaries(
            "一段足够长的测试文本",
            BadLLM(),
            min_chunk_chars=4,
            max_chunk_chars=20,
            max_input_chars=100,
        )

        self.assertEqual(pieces, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_llm_boundary_splitter.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement boundary splitter**

Create `processors/llm_boundary_splitter.py`:

```python
"""Optional LLM-assisted boundary splitter.

The LLM is only allowed to propose boundaries. It must not rewrite source text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class LLMBoundaryPiece:
    text: str
    start_char: int
    end_char: int
    label: str
    reason: str
    strategy: str = "llm_boundary"


_SYSTEM_PROMPT = """你是文档 chunk 边界判断器。只返回 JSON，不要改写原文。
任务：根据主题/步骤/论点变化，给出适合检索的 chunk 边界。
输出格式：{"boundaries":[{"start_char":0,"end_char":123,"label":"...","reason":"..."}]}。
边界必须覆盖输入文本的连续区间，不能重排，不能新增原文没有的内容。"""


def _call_llm(llm: Any, text: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if hasattr(llm, "chat"):
        return llm.chat(messages, temperature=0.0)
    if hasattr(llm, "complete"):
        return llm.complete(messages=messages, temperature=0.0)
    return None


def _parse_response(raw: str) -> List[dict]:
    data = json.loads(raw)
    boundaries = data.get("boundaries")
    if not isinstance(boundaries, list):
        return []
    return [b for b in boundaries if isinstance(b, dict)]


def split_text_with_llm_boundaries(
    text: str,
    llm: Any,
    *,
    min_chunk_chars: int = 300,
    max_chunk_chars: int = 2000,
    max_input_chars: int = 6000,
) -> List[LLMBoundaryPiece]:
    source = (text or "").strip()
    if not source or llm is None or len(source) > max_input_chars:
        return []

    try:
        raw = _call_llm(llm, source)
        if not raw:
            return []
        boundaries = _parse_response(raw)
    except Exception:
        return []

    pieces: List[LLMBoundaryPiece] = []
    previous_end = 0
    for boundary in boundaries:
        start = int(boundary.get("start_char", -1))
        end = int(boundary.get("end_char", -1))
        if start != previous_end or end <= start or end > len(source):
            return []
        size = end - start
        if size < min_chunk_chars or size > max_chunk_chars:
            return []
        pieces.append(LLMBoundaryPiece(
            text=source[start:end],
            start_char=start,
            end_char=end,
            label=str(boundary.get("label") or ""),
            reason=str(boundary.get("reason") or ""),
        ))
        previous_end = end

    if previous_end != len(source):
        return []
    return pieces
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_llm_boundary_splitter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add processors/llm_boundary_splitter.py tests/test_llm_boundary_splitter.py
git commit -m "feat: add llm boundary splitter"
```

### Task 7: Gate LLM Boundary Splitter Behind Low Confidence

**Files:**
- Modify: `processors/chunker.py:1275`
- Modify: `steps/chunk.py`
- Test: `tests/test_llm_boundary_splitter.py`

- [ ] **Step 1: Add parameters to `_maybe_split_text_chunk`**

Add:

```python
    llm: Optional["LLMClient"] = None,
    llm_boundary_enabled: bool = False,
    llm_boundary_only_low_confidence: bool = True,
    llm_boundary_max_input_chars: int = 6000,
```

- [ ] **Step 2: Import splitter**

In `processors/chunker.py`, add:

```python
from .llm_boundary_splitter import split_text_with_llm_boundaries
```

- [ ] **Step 3: Try LLM only after semantic split is needed**

Inside `_maybe_split_text_chunk`, after the existing `semantic_split(...)` call and before returning semantic pieces, add logic equivalent to:

```python
    llm_pieces = []
    if llm_boundary_enabled and llm is not None:
        llm_pieces = split_text_with_llm_boundaries(
            content,
            llm,
            min_chunk_chars=min_chars,
            max_chunk_chars=max_chars,
            max_input_chars=llm_boundary_max_input_chars,
        )
    if llm_pieces:
        pieces = llm_pieces
```

When building sub chunks, support both `.text` from `SemanticChunk` and `LLMBoundaryPiece`; if a piece has `strategy`, use it:

```python
        sub["split_strategy"] = getattr(
            piece,
            "strategy",
            "semantic_embedding" if embedder is not None else "semantic_greedy",
        )
        if hasattr(piece, "label"):
            sub["boundary_label"] = piece.label
            sub["boundary_reason"] = piece.reason
```

- [ ] **Step 4: Pass settings from `build_knowledge_blocks` to `_maybe_split_text_chunk`**

Add parameters to `build_knowledge_blocks`:

```python
    llm_boundary_enabled: bool = False,
    llm_boundary_only_low_confidence: bool = True,
    llm_boundary_max_input_chars: int = 6000,
```

Pass these into `_maybe_split_text_chunk` at call sites.

- [ ] **Step 5: Parse config in `steps/chunk.py`**

In `_run_mineru`, parse:

```python
        llm_boundary_cfg = (cfg.get("llm_boundary") or {}) if isinstance(cfg, dict) else {}
```

Pass:

```python
            llm_boundary_enabled=bool(llm_boundary_cfg.get("enabled", False)),
            llm_boundary_only_low_confidence=bool(llm_boundary_cfg.get("only_low_confidence", True)),
            llm_boundary_max_input_chars=int(llm_boundary_cfg.get("max_input_chars", 6000)),
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
python -m pytest tests/test_llm_boundary_splitter.py tests/test_chunker_diagnostics_integration.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add processors/chunker.py steps/chunk.py tests/test_llm_boundary_splitter.py
git commit -m "feat: gate llm boundary splitting"
```

---

## Phase 4: Optional Section Repair

### Task 8: Add Deterministic Section Candidate Scoring

**Files:**
- Create: `processors/section_repair.py`
- Test: `tests/test_section_repair.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_section_repair.py`:

```python
import unittest

from pipeline.processors.section_repair import score_section_candidate


class TestSectionRepair(unittest.TestCase):
    def test_numbered_short_heading_scores_high(self):
        score = score_section_candidate("2.1 实验方法")
        self.assertGreaterEqual(score.confidence, 0.8)
        self.assertTrue(score.is_likely_title)

    def test_long_sentence_scores_low(self):
        score = score_section_candidate("本文研究耐候钢在海洋大气环境中的腐蚀行为，并分析不同腐蚀产物的形成机制。")
        self.assertLess(score.confidence, 0.5)
        self.assertFalse(score.is_likely_title)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_section_repair.py -v
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement scorer**

Create `processors/section_repair.py`:

```python
"""Section title candidate scoring.

This module starts deterministic. LLM fallback can later be added only for
ambiguous scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBERED_TITLE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+[、.．])\s*\S+")
_SENTENCE_END_RE = re.compile(r"[。！？.!?]\s*$")


@dataclass
class SectionCandidateScore:
    confidence: float
    is_likely_title: bool
    reason: str


def score_section_candidate(text: str) -> SectionCandidateScore:
    value = (text or "").strip()
    if not value:
        return SectionCandidateScore(0.0, False, "empty")

    score = 0.0
    reasons = []

    if _NUMBERED_TITLE_RE.match(value):
        score += 0.55
        reasons.append("numbered")
    if len(value) <= 30:
        score += 0.25
        reasons.append("short")
    if not _SENTENCE_END_RE.search(value):
        score += 0.15
        reasons.append("no_sentence_end")
    if len(value) > 80:
        score -= 0.45
        reasons.append("too_long")
    if _SENTENCE_END_RE.search(value) and len(value) > 30:
        score -= 0.25
        reasons.append("sentence_like")

    confidence = max(0.0, min(1.0, round(score, 4)))
    return SectionCandidateScore(
        confidence=confidence,
        is_likely_title=confidence >= 0.7,
        reason=",".join(reasons) or "weak_signal",
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_section_repair.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add processors/section_repair.py tests/test_section_repair.py
git commit -m "feat: add section candidate scoring"
```

---

## Phase 5: Final Wiring and Regression

### Task 9: Pass Quality Config from Chunk Step

**Files:**
- Modify: `steps/chunk.py:166`
- Test: `tests/test_chunker_diagnostics_integration.py`

- [ ] **Step 1: Parse quality config in `_run_mineru` and `_run_uniparser`**

Add:

```python
        quality_cfg = (cfg.get("quality_gate") or {}) if isinstance(cfg, dict) else {}
```

Pass into chunker calls:

```python
            quality_gate_enabled=bool(quality_cfg.get("enabled", True)),
            quality_min_text_chars=int(quality_cfg.get("min_text_chars", 30)),
            quality_max_noise_ratio=float(quality_cfg.get("max_noise_ratio", 0.35)),
            quality_max_repeated_line_ratio=float(quality_cfg.get("max_repeated_line_ratio", 0.4)),
```

- [ ] **Step 2: Add matching parameters to UniParser chunker**

In `processors/uniparser_chunker.py`, add the same `quality_*` parameters and call `apply_chunk_diagnostics(...)` before return.

- [ ] **Step 3: Run integration tests**

Run:

```bash
python -m pytest tests/test_chunker_diagnostics_integration.py tests/test_mineru_chunker.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add steps/chunk.py processors/uniparser_chunker.py
git commit -m "feat: wire chunk quality config"
```

### Task 10: Regression Test Existing Chunker Behavior

**Files:**
- Existing tests only

- [ ] **Step 1: Run chunker-focused tests**

Run:

```bash
python -m pytest tests/test_mineru_chunker.py tests/test_semantic_splitter_sizing.py tests/test_asset_linker.py tests/test_chunk_diagnostics.py tests/test_llm_boundary_splitter.py tests/test_section_repair.py -v
```

Expected: PASS.

- [ ] **Step 2: Run retrieval tests that depend on chunk metadata**

Run:

```bash
python -m pytest tests/test_structural_retrieval.py tests/test_neighbor_expansion.py tests/test_local_retrieve_anchoring.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit if tests required compatibility fixes**

If code changes were needed:

```bash
git add processors steps tests
git commit -m "fix: preserve chunker regression behavior"
```

If no code changes were needed, do not create an empty commit.

---

## Acceptance Criteria

- Existing default chunk content remains unchanged unless explicit LLM boundary config is enabled.
- Every chunk has `split_strategy`, `chunk_confidence`, `quality_flags`, and `quality_metrics` when `quality_gate.enabled=true`.
- Deterministic asset linking connects text chunks to matching figure/table chunks by explicit number references.
- LLM boundary splitting is disabled by default and returns only source-text slices, never rewritten text.
- LLM failures or malformed JSON responses fall back to existing semantic splitting.
- New tests pass without requiring external network calls.

## Self-Review

- Spec coverage: Covers diagnostics, asset linking, low-confidence LLM boundary fallback, section candidate scoring, config gating, and regression verification.
- Placeholder scan: No implementation placeholders remain; optional future LLM section fallback is explicitly out of implementation scope for this plan.
- Type consistency: Function names and metadata keys are consistent across tasks: `apply_chunk_diagnostics`, `apply_asset_links`, `split_text_with_llm_boundaries`, `score_section_candidate`, `split_strategy`, `chunk_confidence`, `quality_flags`, `quality_metrics`.
