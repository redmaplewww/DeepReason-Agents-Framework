from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import parse, request

from reasoning_agent_template.evidence import EvidenceLedger
from reasoning_agent_template.models import KnowledgeChunk, stable_hash


SUPPORTED_EXTENSIONS = {".md", ".txt", ".json"}
LOCAL_METHODS = {"keyword", "bm25", "semantic", "graph"}
ALL_METHODS = {*LOCAL_METHODS, "wiki"}
METHOD_WEIGHTS = {"keyword": 0.18, "bm25": 0.34, "semantic": 0.32, "graph": 0.16}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}
GLOSSARY = {
    "高熵合金": ["high", "entropy", "alloy", "hea"],
    "合金": ["alloy"],
    "强度": ["strength"],
    "屈服强度": ["yield", "strength"],
    "微观组织": ["microstructure"],
    "组织": ["microstructure"],
    "因素": ["factor", "contributors"],
    "影响": ["influence", "depends"],
    "检索": ["retrieval", "retrieve", "search"],
    "关键词": ["keyword", "lexical", "term"],
    "向量": ["vector", "embedding"],
    "嵌入": ["embedding"],
    "完全匹配": ["exact", "shared", "terms"],
    "不需要": ["without"],
    "长期记忆": ["long", "term", "memory"],
    "记忆": ["memory"],
    "写入": ["write", "writes", "update"],
    "审批": ["approval", "gate"],
    "保护": ["safeguard", "gate"],
    "证据": ["evidence", "support"],
    "不足": ["insufficient", "inadequate", "gap"],
    "状态机": ["state", "machine", "workflow"],
    "回到": ["return", "retry", "back"],
    "环节": ["stage", "node"],
    "自进化": ["self", "evolution", "proposal"],
    "提案": ["proposal", "request"],
    "证据门禁": [
        "evidence",
        "gate",
        "approval",
        "requirements",
        "workspace",
        "boundaries",
    ],
}
SYNONYMS = {
    "recollection": ["memory"],
    "durable": ["long", "term", "persistent"],
    "persistent": ["long", "term"],
    "safeguard": ["gate", "approval"],
    "rollover": ["rotation"],
    "credential": ["secret", "key"],
    "token": ["key", "credential"],
    "finite": ["state"],
    "route": ["workflow", "edge"],
    "support": ["evidence"],
    "inadequate": ["insufficient"],
    "window": ["chunk"],
    "segmentation": ["chunking"],
    "broad": ["large"],
    "neural": ["semantic"],
    "pairwise": ["cross", "encoder"],
    "judge": ["reranker"],
    "passage": ["document", "chunk"],
    "passages": ["documents", "chunks"],
    "compositionally": ["entropy", "alloy"],
    "microstructural": ["microstructure"],
    "contributors": ["factors"],
}


@dataclass(frozen=True)
class _IndexedChunk:
    source: Path
    start_line: int
    end_line: int
    text: str
    content_hash: str


class WikipediaKnowledgeSource:
    """Small Wikipedia API adapter used as an optional fallback retriever."""

    def __init__(
        self,
        *,
        urlopen: Callable[..., Any] | None = None,
        timeout_seconds: int = 4,
        language: str = "en",
        user_agent: str = "reasoning-agent-template/0.1 (local debug console)",
    ):
        self.urlopen = urlopen or request.urlopen
        self.timeout_seconds = timeout_seconds
        self.language = language or "en"
        self.user_agent = user_agent
        self.diagnostics: list[dict[str, Any]] = []

    def retrieve(self, query: str, *, top_k: int = 2) -> list[dict[str, Any]]:
        self.diagnostics = []
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srlimit": str(max(1, top_k)),
            "srsearch": query,
        }
        url = f"https://{self.language}.wikipedia.org/w/api.php?{parse.urlencode(params)}"
        try:
            wiki_request = request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
            )
            response = self.urlopen(wiki_request, timeout=self.timeout_seconds)
            raw = response.read().decode("utf-8")
            rows = json.loads(raw).get("query", {}).get("search", [])
        except Exception as exc:
            self.diagnostics.append(
                {
                    "source": "wikipedia_api",
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            return []
        results: list[dict[str, Any]] = []
        for row in rows[:top_k]:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            snippet = _strip_html(str(row.get("snippet") or ""))
            results.append(
                {
                    "title": title,
                    "url": f"https://{self.language}.wikipedia.org/wiki/{parse.quote(title.replace(' ', '_'))}",
                    "summary": snippet or title,
                    "score": 0.52,
                }
            )
        self.diagnostics.append(
            {
                "source": "wikipedia_api",
                "status": "completed" if results else "empty",
                "message": f"query returned {len(results)} result(s)",
            }
        )
        return results


class LocalKnowledgeBase:
    """Local hybrid RAG index with lexical, semantic, graph, and wiki retrieval."""

    def __init__(
        self,
        root: Path,
        *,
        ledger: EvidenceLedger | None = None,
        wiki_source: Any | None = None,
        max_chunk_chars: int = 1400,
    ):
        self.root = Path(root)
        self.ledger = ledger
        self.wiki_source = wiki_source if wiki_source is not None else WikipediaKnowledgeSource()
        self.max_chunk_chars = max(300, max_chunk_chars)
        self._chunks: list[_IndexedChunk] = []
        self.diagnostics: list[dict[str, Any]] = []

    def ingest(self) -> list[_IndexedChunk]:
        self._chunks = []
        if not self.root.exists():
            return []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                text = self._read_text(path)
                self._chunks.extend(self._chunk_text(path, text))
        return list(self._chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        methods: list[str] | None = None,
        min_score: float = 0.0,
        wiki_top_k: int = 2,
    ) -> list[KnowledgeChunk]:
        self.diagnostics = []
        selected_methods = _normalize_methods(methods)
        if not self._chunks:
            self.ingest()

        local_methods = [method for method in selected_methods if method in LOCAL_METHODS]
        results = self._retrieve_local(query, local_methods=local_methods, min_score=min_score)

        if "wiki" in selected_methods:
            results.extend(self._retrieve_wiki(query, top_k=wiki_top_k, min_score=min_score))

        results = _dedupe_chunks(results)
        results.sort(key=lambda chunk: (-chunk.score, chunk.source, chunk.span))
        return results[: max(1, top_k)]

    def cite(self, chunk: KnowledgeChunk) -> str:
        return f"[{chunk.evidence_id}] {chunk.source}#{chunk.span}"

    def _retrieve_local(self, query: str, *, local_methods: list[str], min_score: float) -> list[KnowledgeChunk]:
        if not self._chunks or not local_methods:
            return []

        raw_query_terms = _terms(query)
        expanded_query_terms = _expanded_terms(query)
        semantic_query = _semantic_vector(query)
        chunk_terms = {chunk: _expanded_terms(chunk.text) for chunk in self._chunks}
        raw_chunk_terms = {chunk: _terms(chunk.text) for chunk in self._chunks}
        semantic_chunks = {chunk: _semantic_vector(chunk.text) for chunk in self._chunks}
        graph = _build_graph(chunk_terms.values())

        method_scores: dict[_IndexedChunk, dict[str, float]] = {chunk: {} for chunk in self._chunks}
        if "keyword" in local_methods:
            for chunk in self._chunks:
                method_scores[chunk]["keyword"] = _overlap_score(raw_query_terms, raw_chunk_terms[chunk])
        if "bm25" in local_methods:
            bm25_scores = _bm25_scores(expanded_query_terms, list(chunk_terms.values()))
            for chunk, score in zip(self._chunks, bm25_scores):
                method_scores[chunk]["bm25"] = score
        if "semantic" in local_methods:
            for chunk in self._chunks:
                method_scores[chunk]["semantic"] = _cosine(semantic_query, semantic_chunks[chunk])
        if "graph" in local_methods:
            for chunk in self._chunks:
                method_scores[chunk]["graph"] = _graph_score(expanded_query_terms, chunk_terms[chunk], graph)

        results: list[KnowledgeChunk] = []
        for chunk in self._chunks:
            breakdown = {method: score for method, score in method_scores[chunk].items() if score > 0}
            if not breakdown:
                continue
            score = _weighted_score(breakdown, local_methods)
            if score < min_score:
                continue
            results.append(self._to_chunk(chunk, score=score, retrieval_method=_method_label(local_methods), breakdown=breakdown))

        self.diagnostics.append(
            {
                "source": "local_rag",
                "status": "completed",
                "message": f"methods={','.join(local_methods) or 'none'} chunks={len(self._chunks)} hits={len(results)}",
            }
        )
        return results

    def _retrieve_wiki(self, query: str, *, top_k: int, min_score: float) -> list[KnowledgeChunk]:
        rows = self.wiki_source.retrieve(query, top_k=top_k)
        self.diagnostics.extend(list(getattr(self.wiki_source, "diagnostics", []) or []))
        chunks: list[KnowledgeChunk] = []
        for row in rows:
            score = float(row.get("score", 0.52))
            if score < min_score:
                continue
            title = str(row.get("title") or "Wikipedia")
            url = str(row.get("url") or "")
            summary = str(row.get("summary") or title)
            evidence_id = ""
            content_hash = stable_hash(f"{url}\n{summary}")
            if self.ledger is not None:
                evidence = self.ledger.record(
                    source_type="wiki",
                    uri=url,
                    locator=title,
                    content=summary,
                    summary=_summarize(summary),
                    confidence=min(1.0, score),
                    used_for=["knowledge:wiki"],
                )
                evidence_id = evidence.id
            chunks.append(
                KnowledgeChunk(
                    source=url,
                    span=title,
                    text=summary,
                    content_hash=content_hash,
                    score=score,
                    evidence_id=evidence_id,
                    retrieval_method="wiki",
                    score_breakdown={"wiki": score},
                    metadata={"title": title},
                )
            )
        self.diagnostics.append(
            {
                "source": "wiki",
                "status": "completed" if chunks else "empty",
                "message": f"wiki returned {len(chunks)} result(s)",
            }
        )
        return chunks

    def _to_chunk(
        self,
        chunk: _IndexedChunk,
        *,
        score: float,
        retrieval_method: str,
        breakdown: dict[str, float],
    ) -> KnowledgeChunk:
        span = f"lines {chunk.start_line}-{chunk.end_line}"
        evidence_id = ""
        if self.ledger is not None:
            evidence = self.ledger.record(
                source_type="file",
                uri=str(chunk.source),
                locator=span,
                content=chunk.text,
                summary=_summarize(chunk.text),
                confidence=min(1.0, 0.45 + score),
                used_for=[f"knowledge:{retrieval_method}"],
            )
            evidence_id = evidence.id
        return KnowledgeChunk(
            source=str(chunk.source),
            span=span,
            text=chunk.text,
            content_hash=chunk.content_hash,
            score=score,
            evidence_id=evidence_id,
            retrieval_method=retrieval_method,
            score_breakdown={key: round(value, 6) for key, value in breakdown.items()},
        )

    def _read_text(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return text
        return text

    def _chunk_text(self, path: Path, text: str) -> list[_IndexedChunk]:
        lines = text.splitlines() or [text]
        chunks: list[_IndexedChunk] = []
        buffer: list[str] = []
        start_line = 1
        current_length = 0

        def flush(end_line: int) -> None:
            nonlocal buffer, start_line, current_length
            cleaned = "\n".join(line for line in buffer if line.strip()).strip()
            if cleaned:
                chunks.append(
                    _IndexedChunk(
                        source=path,
                        start_line=start_line,
                        end_line=max(start_line, end_line),
                        text=cleaned,
                        content_hash=stable_hash(cleaned),
                    )
                )
            buffer = []
            current_length = 0

        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            projected = current_length + len(line) + 1
            if buffer and (projected > self.max_chunk_chars or (not stripped and current_length > self.max_chunk_chars // 2)):
                flush(index - 1)
                start_line = index + 1 if not stripped else index
            if stripped:
                buffer.append(line)
                current_length += len(line) + 1
            elif not buffer:
                start_line = index + 1
        if buffer:
            flush(len(lines))
        return chunks


def _normalize_methods(methods: list[str] | None) -> list[str]:
    if not methods:
        return ["keyword"]
    normalized: list[str] = []
    for method in methods:
        value = str(method).strip().lower()
        if value in {"hybrid", "local-hybrid"}:
            for item in ["bm25", "semantic", "graph"]:
                if item not in normalized:
                    normalized.append(item)
            continue
        if value in {"local-keyword", "keyword"}:
            value = "keyword"
        if value in {"wikipedia", "wiki"}:
            value = "wiki"
        if value in ALL_METHODS and value not in normalized:
            normalized.append(value)
    return normalized or ["keyword"]


def _terms(value: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.update(token)
            terms.update(token[index : index + 2] for index in range(max(0, len(token) - 1)))
        elif len(token) > 2 and token not in STOPWORDS:
            terms.add(token)
    return terms


def _expanded_terms(value: str) -> set[str]:
    terms = set(_terms(value))
    lowered = value.lower()
    for phrase, additions in GLOSSARY.items():
        if phrase in value:
            terms.update(additions)
    for term in list(terms):
        terms.update(SYNONYMS.get(term, []))
        if term.endswith("s") and len(term) > 4:
            terms.add(term[:-1])
        if term.endswith("ing") and len(term) > 6:
            terms.add(term[:-3])
    if "cross-encoder" in lowered or "cross encoder" in lowered:
        terms.update({"cross", "encoder", "reranker"})
    if "high entropy alloy" in lowered:
        terms.update({"high", "entropy", "alloy", "hea"})
    return {term for term in terms if term and term not in STOPWORDS}


def _semantic_vector(value: str) -> Counter[str]:
    terms = _expanded_terms(value)
    vector: Counter[str] = Counter()

    for term in terms:
        contains_cjk = bool(
            re.search(r"[\u4e00-\u9fff]", term)
        )
        vector[term] = 0.25 if contains_cjk else 1.0

    compact = re.sub(r"\s+", " ", value.lower())

    for token in re.findall(
        r"[a-zA-Z][a-zA-Z0-9_-]{3,}",
        compact,
    ):
        token = token.replace("-", "")

        for index in range(max(0, len(token) - 2)):
            vector[f"tri:{token[index:index + 3]}"] += 0.25

    return vector

def _overlap_score(query_terms: set[str], text_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms.intersection(text_terms)) / max(1, len(query_terms))


def _bm25_scores(query_terms: set[str], documents: list[set[str]]) -> list[float]:
    if not query_terms or not documents:
        return [0.0 for _ in documents]
    document_count = len(documents)
    avgdl = sum(len(document) for document in documents) / max(1, document_count)
    df: Counter[str] = Counter()
    for document in documents:
        for term in document:
            df[term] += 1
    k1 = 1.5
    b = 0.75
    raw_scores: list[float] = []
    for document in documents:
        score = 0.0
        frequencies = Counter(document)
        doc_len = max(1, len(document))
        for term in query_terms:
            freq = frequencies.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (document_count - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * ((freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / max(1, avgdl))))
        raw_scores.append(score)
    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _build_graph(documents: Any) -> dict[str, Counter[str]]:
    graph: dict[str, Counter[str]] = defaultdict(Counter)
    for terms in documents:
        filtered = sorted(term for term in terms if term not in STOPWORDS and not term.startswith("tri:"))
        for index, term in enumerate(filtered):
            window = filtered[max(0, index - 8) : index] + filtered[index + 1 : index + 9]
            for neighbor in window:
                if neighbor != term:
                    graph[term][neighbor] += 1
    return graph


def _graph_score(query_terms: set[str], document_terms: set[str], graph: dict[str, Counter[str]]) -> float:
    if not query_terms or not document_terms:
        return 0.0
    expanded: Counter[str] = Counter()
    for term in query_terms:
        expanded[term] += 1.0
        for neighbor, weight in graph.get(term, Counter()).most_common(12):
            expanded[neighbor] += min(0.6, 0.12 * weight)
    if not expanded:
        return 0.0
    score = sum(weight for term, weight in expanded.items() if term in document_terms)
    return min(1.0, score / max(1.0, sum(expanded.values()) * 0.45))


def _weighted_score(breakdown: dict[str, float], methods: list[str]) -> float:
    weights = {method: METHOD_WEIGHTS.get(method, 0.2) for method in methods if method in LOCAL_METHODS}
    denominator = sum(weights.values()) or 1.0
    return sum(breakdown.get(method, 0.0) * weights.get(method, 0.0) for method in weights) / denominator


def _method_label(methods: list[str]) -> str:
    if len(methods) == 1:
        return methods[0]
    return "hybrid:" + "+".join(methods)


def _dedupe_chunks(results: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    best: dict[tuple[str, str], KnowledgeChunk] = {}
    for chunk in results:
        key = (chunk.source, chunk.span)
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk
    return list(best.values())


def _strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _summarize(value: str, limit: int = 180) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
