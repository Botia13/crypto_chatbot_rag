"""Resumable eight-strategy retrieval benchmark. No generator or RAGAS imports.

Models, API clients and the read-existing Qdrant client are initialized lazily.
Gold fields are used only by evaluation, never by the retrieval backend.
"""
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from RAG_model.model_analysis_notebooks.utils.evidence import (
    EVIDENCE_METRIC_VERSION, evaluate_claim_evidence, load_evaluation_questions,
    normalize_evidence_accession,
)

STRATEGIES = (
    {"id": 1, "label": "1. Unfiltered hybrid baseline", "kind": "simple", "filter": "none", "cap": 75},
    {"id": 2, "label": "2. Ticker filtering", "kind": "simple", "filter": "ticker", "cap": 75},
    {"id": 3, "label": "3. Ticker + reporting-period filtering", "kind": "simple", "filter": "period", "cap": 75},
    {"id": 5, "label": "5. SEC rewrite + ticker + reporting-period", "kind": "rewrite", "filter": "period", "cap": 75},
    {"id": 9, "label": "9. Ticker-filtered variable subqueries", "kind": "facets", "filter": "ticker", "cap": 75},
    {"id": 15, "label": "15. Rewrite + ticker variable subqueries", "kind": "rewrite_facets", "filter": "ticker", "cap": 75},
    {"id": 18, "label": "18. Legacy fixed-depth facets @50", "kind": "legacy", "filter": "ticker", "cap": 50},
    {"id": 20, "label": "20. Legacy confidence backfill @75", "kind": "routed", "filter": "ticker", "cap": 75},
)
DEFAULT_SETTINGS = {
    "embedding_model": "openai/text-embedding-3-small",
    "planner_model": "openai/gpt-5.6-luna", "planner_temperature": 0,
    "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_max_length": 1500,
    "reranker_batch_size": 16, "max_facets": 24, "variable_depth": 10,
    "depths": [10, 20, 30, 40, 50], "rrf_constant": 60,
    "routing_depth": 10, "routing_top_accessions": 2, "routing_threshold": 0.60,
}
COLLECTION = "sec_filings__chunk-500__overlap-120__encoding-cl100k_base__embedding-openai-text-embedding-3-small"
SCOPE_PATTERN = re.compile(r"\b(IBIT|ETHA|FBTC|FETH|GBTC|ETHE)\s+(10-K|10-Q)\s+for the reporting period ending\s+(\d{4}-\d{2}-\d{2})")
STAGES = ("pool", "raw_15", "raw_20", "raw_50", "raw_75", "metadata_15", "metadata_20")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DiskCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.events = []

    def path(self, stage, key):
        return self.directory / stage / (digest(key) + ".json")

    def get(self, stage, key, compute):
        path = self.path(stage, key)
        start = time.perf_counter()
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["key_digest"] != digest(key):
                raise ValueError(f"Invalid cache key in {path}")
            hit = True
        else:
            value = compute()  # Failures are never persisted as successful entries.
            record = {"key_digest": digest(key), "value": value,
                      "compute_seconds": time.perf_counter() - start,
                      "created_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(path, record)
            hit = False
        elapsed = time.perf_counter() - start
        self.events.append({"stage": stage, "hit": hit, "elapsed_seconds": elapsed})
        return record["value"], {"cache_hit": hit, "wall_seconds": elapsed,
                                  "saved_compute_seconds": record["compute_seconds"]}


def source_scope(question):
    """Read explicit, public question text, never reference evidence or gold IDs."""
    marker = "Source scope:"
    if marker not in question:
        raise ValueError("Use the scoped benchmark: question is missing Source scope.")
    scopes = {}
    for ticker, form, period in SCOPE_PATTERN.findall(question.split(marker, 1)[1]):
        scopes.setdefault(ticker, []).append({"form_type": form, "period_end": period})
    if not scopes:
        raise ValueError("No valid ticker/form/period source scope in question.")
    return scopes


def filter_spec(ticker=None, scopes=(), accessions=()):
    return {"ticker": ticker, "scopes": list(scopes), "accessions": list(accessions)}


def make_filter(spec):
    from qdrant_client import models as m
    conditions = []
    if spec.get("ticker"):
        conditions.append(m.FieldCondition(key="ticker", match=m.MatchValue(value=spec["ticker"])))
    if spec.get("accessions"):
        conditions.append(m.FieldCondition(key="accession_number", match=m.MatchAny(any=spec["accessions"])))
    if spec.get("scopes"):
        conditions.append(m.Filter(should=[m.Filter(must=[
            m.FieldCondition(key="form_type", match=m.MatchValue(value=s["form_type"])),
            m.FieldCondition(key="period_end", match=m.MatchValue(value=s["period_end"])),
        ]) for s in spec["scopes"]]))
    return m.Filter(must=conditions) if conditions else None


def assert_scope(chunks, spec):
    for chunk in chunks:
        valid = (not spec.get("ticker") or chunk.get("ticker") == spec["ticker"])
        valid &= (not spec.get("accessions") or chunk.get("accession_number") in spec["accessions"])
        valid &= (not spec.get("scopes") or any(
            chunk.get("form_type") == s["form_type"] and chunk.get("period_end") == s["period_end"]
            for s in spec["scopes"]))
        if not valid:
            raise AssertionError(f"Filter violation: {chunk.get('chunk_id')} does not satisfy {spec}")


def hybrid_search(client, collection, dense, sparse, spec, limit):
    from qdrant_client import models as m
    query_filter = make_filter(spec)
    result = client.query_points(collection_name=collection, prefetch=[
        m.Prefetch(query=dense, using="dense", limit=limit, filter=query_filter),
        m.Prefetch(query=m.SparseVector(**sparse), using="bm25", limit=limit, filter=query_filter),
    ], query=m.FusionQuery(fusion=m.Fusion.RRF), query_filter=query_filter,
       limit=limit, with_payload=True)
    chunks = []
    for point in result.points:
        chunk = dict(point.payload or {})
        chunk["section_title"] = chunk.get("section_title") or chunk.get("chunk_title")
        chunk["score"] = float(point.score)
        chunks.append(chunk)
    assert_scope(chunks, spec)
    return chunks


def metadata_passage(chunk):
    fields = (("Ticker", "ticker"), ("Accession", "accession_number"),
              ("Filing date", "filing_date"), ("Reporting period", "period_end"),
              ("Section", "section_title"), ("Section key", "section_key"), ("Table", "table_title"))
    return "\n".join(f"{name}: {chunk.get(key) or 'N/A'}" for name, key in fields) + "\n\nContent:\n" + str(chunk.get("chunk_text") or "")


class LocalBackend:
    """No initialization here performs inference, network access or opens the DB."""
    def __init__(self, root, settings):
        self.root, self.settings = Path(root), settings
        self._api = self._bm25 = self._qdrant = self._reranker = None

    @property
    def api(self):
        if self._api is None:
            from dotenv import load_dotenv
            from RAG_model.ingestion.embedding import create_openrouter_client
            load_dotenv(self.root / ".env")
            self._api = create_openrouter_client()
        return self._api

    def chat(self, prompt, max_tokens):
        response = self.api.chat.completions.create(model=self.settings["planner_model"],
            temperature=self.settings["planner_temperature"], max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        text = response.choices[0].message.content or ""
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I))

    def embed(self, text):
        response = self.api.embeddings.create(model=self.settings["embedding_model"], input=[text])
        if self._bm25 is None:
            from fastembed import SparseTextEmbedding
            self._bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
        sparse = next(self._bm25.embed(text))
        return {"dense": response.data[0].embedding,
                "sparse": {"indices": sparse.indices.tolist(), "values": sparse.values.tolist()}}

    def search(self, collection, vector, spec, limit):
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(path=str(self.root / "data/qdrant_storage"))
        return hybrid_search(self._qdrant, collection, vector["dense"], vector["sparse"], spec, limit)

    def rerank(self, question, passages):
        if not passages:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.settings["reranker_model"], max_length=self.settings["reranker_max_length"])
        return [float(s) for s in self._reranker.predict([(question, p) for p in passages],
            batch_size=self.settings["reranker_batch_size"], show_progress_bar=False)]

    def close(self):
        if self._qdrant is not None:
            self._qdrant.close()
            self._qdrant = None


def round_robin(groups):
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


def deduplicate(chunks, limit):
    unique = {}
    for chunk in chunks:
        identity = chunk["chunk_id"]
        if identity not in unique:
            unique[identity] = dict(chunk)
            unique[identity]["retrieval_provenance"] = list(chunk.get("retrieval_provenance", []))
        else:
            for p in chunk.get("retrieval_provenance", []):
                if p not in unique[identity]["retrieval_provenance"]:
                    unique[identity]["retrieval_provenance"].append(p)
    return list(unique.values())[:limit]


def fuse(groups, limit, constant):
    scores, rank = defaultdict(float), {}
    for group in groups:
        for i, c in enumerate(group, 1):
            scores[c["chunk_id"]] += 1 / (constant + i)
            rank[c["chunk_id"]] = min(rank.get(c["chunk_id"], i), i)
    chunks = deduplicate([c for g in groups for c in g], sum(map(len, groups)))
    return sorted(chunks, key=lambda c: (-scores[c["chunk_id"]], rank[c["chunk_id"]], c["chunk_id"]))[:limit]


class Benchmark:
    def __init__(self, root, *, settings=None, backend=None, cache_dir=None,
                 collection=COLLECTION, collection_identity=None):
        self.root = Path(root)
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.collection = collection
        self.backend = backend or LocalBackend(root, self.settings)
        self.cache = DiskCache(cache_dir or self.root / "data/rag_evaluation/results/retrieval_benchmark/cache")
        self.implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if collection_identity is None:
            db = self.root / "data/qdrant_storage/collection" / collection / "storage.sqlite"
            stat = db.stat()  # Fail clearly if the existing collection is missing; never reindex.
            from importlib.metadata import version
            collection_identity = {"name": collection, "path": str(db.resolve()), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "qdrant_client": version("qdrant-client"),
                "meta": digest((self.root / "data/qdrant_storage/meta.json").read_text())}
        self.identity = collection_identity

    def key(self, **values):
        return {"implementation": self.implementation, **values}

    @property
    def retrieval_settings(self):
        return {k: v for k, v in self.settings.items() if not k.startswith("reranker_")}

    def pool_key(self, question, strategy):
        return self.key(question=question, strategy=strategy, settings=self.retrieval_settings, collection=self.identity)

    def rewrite(self, question):
        prompt = ("Rewrite the user question as one retrieval query aligned with SEC filing vocabulary. "
                  "Preserve every ticker, entity, year, reporting period, form type, and requested fact. "
                  "Do not answer or add facts. Return JSON only: {\"rewritten_question\": \"...\"}.\nUSER QUESTION:\n" + question)
        def compute():
            result = self.backend.chat(prompt, 1200)
            value = str(result.get("rewritten_question", "")).strip()
            original_scope = source_scope(question)
            if len(value) < 10 or any(t not in value.upper() for t in original_scope):
                raise ValueError("Rewrite dropped a ticker or returned empty content; retry after inspecting failure.")
            # Keep the explicit source contract byte-identical even if the rewrite paraphrases it.
            value = value.split("Source scope:", 1)[0].rstrip() + "\nSource scope:" + question.split("Source scope:", 1)[1]
            return {"rewritten_question": value, "fallback_used": False}
        return self.cache.get("rewrite", self.key(prompt=prompt, model=self.settings["planner_model"],
            temperature=self.settings["planner_temperature"]), compute)[0]["rewritten_question"]

    def plan(self, question, legacy=False):
        scopes = source_scope(question)
        preface = ("Create a complete ticker-by-facet retrieval checklist for the SEC-filing question. "
            "For every requested attribute, create one separate atomic facet for every applicable ticker. "
            "Two tickers and four attributes require eight facets. Do not create a single comparison facet "
            "when both sides can be retrieved independently. For synthesis, explicitly separate every "
            "operational, structural, financial, or procedural dimension. " if legacy else
            "Decompose the question into complete, atomic SEC-filing retrieval facets. "
            "Represent every requested attribute for every applicable ticker. ")
        prompt = (preface + "Preserve dates, reporting periods, form types, entities, and technical terms. "
            "Never combine multiple requested facts in one facet. Do not answer or invent values or accessions. "
            f"Return no more than {self.settings['max_facets']} facets. Allowed tickers: {', '.join(scopes)}.\n"
            'Return JSON only: {"facets":[{"ticker":"TICKER","facet":"atomic attribute",'
            '"semantic_query":"natural-language query","filing_query":"SEC-style query"}]}\nQuestion: ' + question)
        def compute():
            result = self.backend.chat(prompt, 6000)
            facets, seen = [], set()
            for raw in result.get("facets", []):
                ticker = str(raw.get("ticker", "")).upper().strip()
                facet = str(raw.get("facet", "")).strip()
                key = ticker, re.sub(r"\W+", " ", facet.lower()).strip()
                if ticker not in scopes or not facet or key in seen:
                    continue
                if not all(str(raw.get(k, "")).strip() for k in ("semantic_query", "filing_query")):
                    continue
                seen.add(key)
                facets.append({"ticker": ticker, "facet": facet, "facet_id": f"facet-{len(facets)+1:02d}",
                    "semantic_query": raw["semantic_query"].strip(), "filing_query": raw["filing_query"].strip()})
                if len(facets) == self.settings["max_facets"]:
                    break
            if {f["ticker"] for f in facets} != set(scopes):
                raise ValueError("Invalid facet plan: missing ticker coverage; no silent fallback.")
            return {"facets": facets, "fallback_used": False}
        return self.cache.get("plan", self.key(prompt=prompt, model=self.settings["planner_model"],
            temperature=self.settings["planner_temperature"]), compute)[0]

    def search(self, query, spec, depth, provenance):
        vector, _ = self.cache.get("embedding", self.key(query=query, model=self.settings["embedding_model"],
            sparse_model="Qdrant/bm25"), lambda: self.backend.embed(query))
        chunks, _ = self.cache.get("search", self.key(query=query, vector=digest(vector), spec=spec,
            depth=depth, collection=self.identity),
            lambda: self.backend.search(self.collection, vector, spec, depth))
        assert_scope(chunks, spec)  # Also check cached search results.
        return [{**c, "retrieval_provenance": [{**provenance, "query": query, "rank": i,
                "depth": depth, "filter": spec}]} for i, c in enumerate(chunks, 1)]

    def route(self, facets):
        def compute():
            scores = defaultdict(lambda: defaultdict(float))
            votes, totals = defaultdict(Counter), Counter()
            for facet in facets:
                ticker = facet["ticker"]
                for kind in ("semantic_query", "filing_query"):
                    chunks = self.search(facet[kind], filter_spec(ticker), self.settings["routing_depth"],
                        {"facet_id": facet["facet_id"], "query_kind": kind, "purpose": "routing"})
                    totals[ticker] += 1
                    seen = set()
                    for rank, chunk in enumerate(chunks, 1):
                        accession = chunk["accession_number"]
                        if accession not in seen:
                            if not seen:
                                votes[ticker][accession] += 1
                            scores[ticker][accession] += 1 / (self.settings["rrf_constant"] + rank)
                            seen.add(accession)
            selected, confidence = {}, {}
            for ticker in totals:
                confidence[ticker] = max(votes[ticker].values(), default=0) / totals[ticker]
                ordered = sorted(scores[ticker], key=lambda a: (-scores[ticker][a], a))
                selected[ticker] = ordered[:self.settings["routing_top_accessions"]] if confidence[ticker] >= self.settings["routing_threshold"] else []
            return {"selected_accessions": selected, "routing_confidence": confidence}
        return self.cache.get("routing", self.key(facets=facets, settings=self.retrieval_settings,
            collection=self.identity), compute)[0]

    def retrieve(self, question, strategy):
        """Only a question string and strategy enter retrieval; no gold rows."""
        def compute():
            scope = source_scope(question)
            kind, cap = strategy["kind"], strategy["cap"]
            diagnostics = {"fallback_used": False}
            if kind in {"simple", "rewrite"}:
                query = self.rewrite(question) if kind == "rewrite" else question
                specs = [filter_spec()] if strategy["filter"] == "none" else [
                    filter_spec(t, periods if strategy["filter"] == "period" else ()) for t, periods in scope.items()]
                groups = [self.search(query, spec, math.ceil(cap / len(specs)),
                    {"purpose": "candidate", "facet_id": "full_question", "query_kind": "full_question"}) for spec in specs]
                candidates = deduplicate(round_robin(groups), cap)
            else:
                query = self.rewrite(question) if kind == "rewrite_facets" else question
                plan = self.plan(query, legacy=kind in {"legacy", "routed"})
                facets = plan["facets"]
                diagnostics["plan"] = plan
                route = self.route(facets) if kind == "routed" else {"selected_accessions": {}}
                diagnostics["route"] = route
                depths = self.settings["depths"] if kind == "routed" else [self.settings["variable_depth"]]
                diagnostics["attempted_depths"] = []
                for depth in depths:
                    pools = []
                    for facet in facets:
                        spec = filter_spec(facet["ticker"], accessions=route["selected_accessions"].get(facet["ticker"], []))
                        kinds = ("semantic_query", "filing_query") if kind in {"legacy", "routed"} else ("filing_query",)
                        groups = [self.search(facet[k], spec, depth, {"purpose": "candidate",
                            "facet_id": facet["facet_id"], "query_kind": k}) for k in kinds]
                        pools.append(fuse(groups, depth, self.settings["rrf_constant"]) if len(groups) == 2 else groups[0])
                    candidates = deduplicate(round_robin(pools), cap)
                    diagnostics["attempted_depths"].append(depth)
                    if len(candidates) >= cap:
                        break
            diagnostics["target_reached"] = len(candidates) >= cap
            return {"candidates": candidates, "diagnostics": diagnostics}
        return self.cache.get("pool", self.pool_key(question, strategy), compute)

    def rerank(self, question, candidates):
        # Canonicalize inputs so identical pools in different orders share inference.
        ordered = sorted(candidates, key=lambda c: c["chunk_id"])
        key = self.key(question=question, passages=[(c["chunk_id"], metadata_passage(c)) for c in ordered],
            model=self.settings["reranker_model"], max_length=self.settings["reranker_max_length"],
            batch_size=self.settings["reranker_batch_size"])
        def compute():
            values = self.backend.rerank(question, [metadata_passage(c) for c in ordered])
            if len(values) != len(ordered) or not all(math.isfinite(v) for v in values):
                raise ValueError("Reranker returned missing or nonfinite scores.")
            return {c["chunk_id"]: float(s) for c, s in zip(ordered, values)}
        scores, timing = self.cache.get("rerank", key, compute)
        ranked = sorted(candidates, key=lambda c: (-scores[c["chunk_id"]], c["chunk_id"]))
        return [{**c, "metadata_rerank_score": scores[c["chunk_id"]]} for c in ranked], timing

    def run(self, questions, *, output_dir=None, progress=print):
        rows = [r for r in questions.to_dict("records") if str(r["answerable"]).lower() in {"true", "1"}]
        if not rows:
            raise ValueError("No answerable questions.")
        if len({r['question_id'] for r in rows}) != len(rows):
            raise ValueError("Duplicate question IDs.")
        metric_code = Path(__file__).with_name("evidence.py").read_bytes()
        manifest = {"dataset_digest": digest(rows), "collection": self.identity, "settings": self.settings,
            "implementation": self.implementation, "metric_version": EVIDENCE_METRIC_VERSION,
            "metric_code_digest": hashlib.sha256(metric_code).hexdigest(), "strategies": STRATEGIES}
        run_id = digest(manifest)[:20]
        directory = Path(output_dir or self.root / "data/rag_evaluation/results/retrieval_benchmark/runs") / run_id
        directory.mkdir(parents=True, exist_ok=True)
        manifest.update(run_id=run_id, status="running", started_at=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / "manifest.json", manifest)
        hits = sum(self.cache.path("pool", self.pool_key(q["question"], s)).exists() for q in rows for s in STRATEGIES)
        progress(f"Candidate pools cached: {hits}/{len(rows)*len(STRATEGIES)}. Missing: {len(rows)*len(STRATEGIES)-hits}. Metadata rankings are checked per pool.")
        question_rows, claim_rows, evidence_rows, failures = [], [], [], []
        try:
            for q in rows:
                for strategy in STRATEGIES:
                    context = {"question_id": q["question_id"], "category": q["category"],
                               "strategy_id": strategy["id"], "strategy": strategy["label"]}
                    started = time.perf_counter()
                    try:
                        progress(f"Q{q['question_id']} experiment {strategy['id']}: checking retrieval and ranking caches...")
                        pool, retrieval_time = self.retrieve(q["question"], strategy)
                        candidates = pool["candidates"]
                        ranked, rerank_time = self.rerank(q["question"], candidates)
                        stages = {"pool": candidates, **{f"raw_{k}": candidates[:k] for k in (15,20,50,75)
                            if k <= strategy["cap"]}, "metadata_15": ranked[:15], "metadata_20": ranked[:20]}
                        evaluation_key = {"manifest": manifest["metric_code_digest"], "question": q,
                            "stages": stages, "metric_version": EVIDENCE_METRIC_VERSION}
                        def evaluate():
                            claims_out, evidence_out = [], []
                            for stage, chunks in stages.items():
                                for name, adjacent in (("corrected", True), ("legacy_single", False)):
                                    claims, evidence = evaluate_claim_evidence(q["required_claims"], q["reference_evidence"], chunks, allow_adjacent=adjacent)
                                    claims_out.extend({"stage": stage, "metric": name, **c} for c in claims)
                                    evidence_out.extend({"stage": stage, "metric": name, **e} for e in evidence)
                            return {"claims": claims_out, "evidence": evidence_out}
                        evaluation, _ = self.cache.get("evaluation", evaluation_key, evaluate)
                        claim_rows.extend({**context, **r} for r in evaluation["claims"])
                        evidence_rows.extend({**context, **r} for r in evaluation["evidence"])
                        question_rows.append({**context, "candidate_count": len(candidates),
                            "candidate_limit": strategy["cap"], "underfilled": len(candidates)<strategy["cap"],
                            "retrieval_cache_hit": retrieval_time["cache_hit"], "rerank_cache_hit": rerank_time["cache_hit"],
                            "retrieval_saved_seconds": retrieval_time["saved_compute_seconds"],
                            "rerank_saved_seconds": rerank_time["saved_compute_seconds"],
                            "cache_load_seconds": sum(t["wall_seconds"] for t in (retrieval_time, rerank_time) if t["cache_hit"]),
                            "wall_seconds_this_run": time.perf_counter()-started})
                        atomic_json(directory / "questions" / f"q{q['question_id']}_experiment{strategy['id']}.json",
                            {**context, "question": q["question"], **pool, "metadata_ranked": ranked,
                             "evaluation": evaluation, "retrieval_timing": retrieval_time, "rerank_timing": rerank_time})
                        progress(f"{len(question_rows)}/{len(rows)*len(STRATEGIES)}: Q{q['question_id']} experiment {strategy['id']} — {len(candidates)} candidates; pool {'cached' if retrieval_time['cache_hit'] else 'computed'}, rerank {'cached' if rerank_time['cache_hit'] else 'computed'}")
                    except Exception as error:
                        failure = {**context, "error": f"{type(error).__name__}: {error}"}
                        failures.append(failure)
                        atomic_json(directory / "failures.json", failures)
                        raise RuntimeError(f"Q{q['question_id']} experiment {strategy['id']} failed. Completed work is cached. {error}") from error
            qdf, cdf, edf = pd.DataFrame(question_rows), pd.DataFrame(claim_rows), pd.DataFrame(evidence_rows)
            overall, category = summary_tables(qdf, cdf)
            # Exports are regenerable from atomic per-question checkpoints and stage caches.
            for name, frame in (("question_results", qdf), ("claim_results", cdf), ("evidence_results", edf),
                                ("overall_results", overall), ("category_results", category)):
                frame.to_csv(directory / f"{name}.csv", index=False)
            atomic_json(directory / "failures.json", [])
            manifest.update(status="complete", completed_runs=len(question_rows),
                            completed_at=datetime.now(timezone.utc).isoformat(), cache_events=self.cache.events)
            atomic_json(directory / "manifest.json", manifest)
            progress(f"RETRIEVAL_BENCHMARK_SAVE_COMPLETE: {directory}")
            return {"overall": overall, "category": category, "questions": qdf, "claims": cdf,
                    "evidence": edf, "directory": directory, "manifest": manifest}
        except BaseException:
            manifest.update(status="incomplete", completed_runs=len(question_rows), failures=failures)
            atomic_json(directory / "manifest.json", manifest)
            raise
        finally:
            self.backend.close()


def summary_tables(question_rows, claim_rows):
    corrected = claim_rows[claim_rows["metric"].eq("corrected")]
    indexes = ["strategy_id", "strategy"]
    def aggregate(keys):
        recalls = corrected.groupby(keys+["stage"])["claim_complete"].mean().unstack("stage")
        recalls = recalls.reindex(columns=STAGES).rename(columns={s: f"claim_recall_{s}" for s in STAGES})
        counts = question_rows.groupby(keys).agg(mean_candidate_count=("candidate_count","mean"),
            underfilled_questions=("underfilled","sum"), mean_retrieval_saved_seconds=("retrieval_saved_seconds","mean"),
            mean_rerank_saved_seconds=("rerank_saved_seconds","mean"), mean_cache_load_seconds=("cache_load_seconds","mean"),
            mean_wall_seconds_this_run=("wall_seconds_this_run","mean"))
        return recalls.join(counts).reset_index().sort_values(keys)
    overall, category = aggregate(indexes), aggregate(indexes+["category"])
    for stage in ("pool", "metadata_15", "metadata_20"):
        stats = category.groupby("strategy_id")[f"claim_recall_{stage}"].agg(["mean","min"])
        overall[f"category_mean_{stage}"] = overall["strategy_id"].map(stats["mean"])
        overall[f"category_min_{stage}"] = overall["strategy_id"].map(stats["min"])
    return overall, category
