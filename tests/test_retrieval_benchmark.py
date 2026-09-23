"""Offline tests: no OpenRouter, model downloads, inference, or real DB writes."""
import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient, models as m

from RAG_model.retrieval.evidence import (
    best_evidence_match, calculate_claim_evidence_recall, load_evaluation_questions,
)
from RAG_model.retrieval import retrieval_benchmark as rb

ROOT = Path(__file__).resolve().parents[1]
ACC = "0000950170-25-039374"


def chunk(index, text, accession=ACC, section="item-1", **extra):
    return {"chunk_id": f"{accession}*{section}*text-{index:04d}", "chunk_text": text,
            "accession_number": accession, "section_key": section, "ticker": "FETH",
            "form_type": "10-K", "period_end": "2024-12-31", "score": 1.0, **extra}


def test_adjacent_support_is_strict_and_requires_both_selected_chunks():
    evidence = {"accession": ACC, "claim_index": 1,
                "supporting_passage": "alpha bravo charlie delta echo foxtrot"}
    first, second = chunk(0, "alpha bravo charlie"), chunk(1, "delta echo foxtrot")
    assert best_evidence_match(evidence, [first, second])["matched"]
    assert best_evidence_match(evidence, [second, first])["matched"]
    assert best_evidence_match(evidence, [first, second])["support_type"] == "adjacent_pair"
    assert not best_evidence_match(evidence, [first])["matched"]
    assert not best_evidence_match(evidence, [first, second], allow_adjacent=False)["matched"]
    for other in [chunk(2, second["chunk_text"]), chunk(1, second["chunk_text"], section="item-2"),
                  chunk(1, second["chunk_text"], accession="0000950170-25-039452"),
                  {**second, "chunk_type": "table"}, {**second, "section_key": "item-2"}]:
        assert not best_evidence_match(evidence, [first, other])["matched"]


def test_claim_requires_all_annotated_evidence_and_missing_annotation_fails():
    evidence = [{"accession": ACC, "claim_index": 1, "supporting_passage": "alpha bravo"},
                {"accession": ACC, "claim_index": 1, "supporting_passage": "november whiskey"}]
    assert calculate_claim_evidence_recall(["claim"], evidence, [chunk(0,"alpha bravo")]) == 0
    assert calculate_claim_evidence_recall(["claim", "unannotated"], evidence,
        [chunk(0,"alpha bravo november whiskey")]) == .5
    assert calculate_claim_evidence_recall([], [], []) is None
    assert not best_evidence_match(evidence[0], [chunk(0,"alpha bravo", accession="0000000")])["matched"]


@pytest.fixture
def memory_index():
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config={"dense": m.VectorParams(size=2,distance=m.Distance.COSINE)},
                             sparse_vectors_config={"bm25": m.SparseVectorParams()})
    specs = [("FETH", "2024-12-31", "10-K", "a"), ("FETH", "2025-12-31", "10-K", "b"),
             ("FBTC", "2024-12-31", "10-K", "c"), ("FETH", "2024-12-31", "10-Q", "d")]
    client.upsert("test", points=[m.PointStruct(id=i, vector={"dense": [1.,float(i)/10],
        "bm25": m.SparseVector(indices=[1],values=[1.])},payload={"ticker":t,"period_end":p,
        "form_type":f,"accession_number":a,"chunk_id":str(i),"chunk_text":"test"})
        for i,(t,p,f,a) in enumerate(specs)])
    yield client
    client.close()


@pytest.mark.parametrize("spec,expected", [
    (rb.filter_spec(), {"0","1","2","3"}),
    (rb.filter_spec("FETH"), {"0","1","3"}),
    (rb.filter_spec("FETH", [{"form_type":"10-K","period_end":"2024-12-31"}]), {"0"}),
    (rb.filter_spec("FETH", accessions=["b"]), {"1"}),
    (rb.filter_spec("FETH", accessions=["missing"]), set()),
    (rb.filter_spec("FBTC", [{"form_type":"10-K","period_end":"2024-12-31"}]), {"2"}),
])
def test_dense_sparse_fusion_filters(memory_index, spec, expected):
    result = rb.hybrid_search(memory_index, "test", [1.,0.], {"indices":[1],"values":[1.]}, spec, 10)
    assert {c["chunk_id"] for c in result} == expected


def test_invalid_cached_results_are_not_silently_filtered():
    with pytest.raises(AssertionError, match="Filter violation"):
        rb.assert_scope([chunk(0,"text")], rb.filter_spec("FBTC"))


def test_scoped_dataset_preserves_labels_and_has_entity_specific_periods():
    original = load_evaluation_questions(ROOT / "data/rag_evaluation/evaluation_questions_v3.csv")
    scoped = load_evaluation_questions(ROOT / "data/rag_evaluation/evaluation_questions_v4.csv")
    assert len(scoped) == 40
    for a,b in zip(original.to_dict("records"), scoped.to_dict("records")):
        assert b["original_question"] == a["question"]
        for name in ("question_id", "category", "required_claims", "reference_evidence"):
            assert a[name] == b[name]
        if str(a["answerable"]).lower() not in {"true","1"}:
            assert a["question"] == b["question"]
        else:
            assert rb.source_scope(b["question"])
            assert all(e["accession"] not in b["question"] for e in b["reference_evidence"])
    scope = rb.source_scope(scoped.loc[scoped.question_id.eq(19), "question"].iloc[0])
    assert scope["GBTC"] == [{"form_type":"10-K","period_end":"2024-12-31"}]
    assert scope["IBIT"] == [{"form_type":"10-Q","period_end":"2025-03-31"}]


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.forbid = False

    def record(self, name):
        assert not self.forbid, f"Unexpected uncached model/search call: {name}"
        self.calls.append(name)

    def chat(self, prompt, max_tokens):
        self.record("chat")
        if "rewritten_question" in prompt:
            return {"rewritten_question":prompt.split("USER QUESTION:\n",1)[1]}
        question = prompt.split("Question: ",1)[1]
        return {"facets":[{"ticker":t,"facet":"fee","semantic_query":t+" fee",
                           "filing_query":t+" Sponsor Fee"} for t in rb.source_scope(question)]}

    def embed(self, text):
        self.record("embed")
        return {"dense":[1.,0.],"sparse":{"indices":[1],"values":[1.]}}

    def search(self, collection, vector, spec, limit):
        self.record("search")
        ticker = spec["ticker"] or "FETH"
        scope = (spec["scopes"] or [{"form_type":"10-K","period_end":"2024-12-31"}])[0]
        accession = (spec["accessions"] or [ACC])[0]
        return [chunk(i, "alpha bravo" if i%2==0 else "charlie delta", accession=accession,
                      ticker=ticker, **scope) for i in range(limit)]

    def rerank(self, question, passages):
        self.record("rerank")
        if self.fail:
            raise RuntimeError("Simulated interruption")
        return [float(len(passages)-i) for i in range(len(passages))]

    def close(self):
        pass


def small_questions():
    return pd.DataFrame([{"question_id":i,"question":f"Question {i}: what is the fee?\nSource scope: FETH 10-K for the reporting period ending 2024-12-31.",
        "category":category,"answerable":True,"required_claims":["all words"],
        "reference_evidence":[{"claim_index":1,"accession":ACC,"supporting_passage":"alpha bravo charlie delta"}]}
        for i,category in enumerate(["Factual","Procedural","Comparison","Synthesis"],1)])


def engine(tmp_path, backend, **kwargs):
    return rb.Benchmark(ROOT,backend=backend,cache_dir=tmp_path/"cache",collection_identity="test-index",**kwargs)


def test_all_strategies_complete_and_full_cache_replay(tmp_path):
    backend = FakeBackend()
    result = engine(tmp_path, backend).run(small_questions(), output_dir=tmp_path/"runs", progress=lambda s:None)
    assert list(result["overall"].strategy_id) == [1,2,3,5,9,15,18,20]
    assert len(result["category"]) == 32
    assert len(result["questions"]) == 32
    assert result["overall"].loc[result["overall"].strategy_id.eq(18), "claim_recall_raw_75"].isna().all()
    assert result["manifest"]["status"] == "complete"
    assert result["questions"].query("strategy_id == 18").candidate_count.max() <= 50
    assert result["claims"].query("metric == 'corrected'").claim_complete.all()
    assert not result["claims"].query("metric == 'legacy_single'").claim_complete.any()
    backend2 = FakeBackend(); backend2.forbid = True
    replay = engine(tmp_path, backend2).run(small_questions(),output_dir=tmp_path/"runs",progress=lambda s:None)
    assert replay["questions"].retrieval_cache_hit.all()
    assert replay["questions"].rerank_cache_hit.all()
    assert replay["directory"] == result["directory"]
    pd.testing.assert_frame_equal(replay["claims"],result["claims"])


def test_interrupted_rerank_preserves_pool_and_resumes(tmp_path):
    backend = FakeBackend(); backend.fail = True
    with pytest.raises(RuntimeError, match="Completed work is cached"):
        engine(tmp_path,backend).run(small_questions().head(1),output_dir=tmp_path/"runs",progress=lambda s:None)
    manifest = json.loads(next((tmp_path/"runs").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "incomplete"
    assert list((tmp_path/"cache/pool").glob("*.json"))
    assert not list((tmp_path/"cache/rerank").glob("*.json"))
    backend.fail = False
    result = engine(tmp_path,backend).run(small_questions().head(1),output_dir=tmp_path/"runs",progress=lambda s:None)
    assert result["questions"].iloc[0].retrieval_cache_hit
    assert result["manifest"]["status"] == "complete"


def test_evidence_only_change_reuses_models_but_changes_evaluation(tmp_path):
    q = small_questions().head(1)
    first = engine(tmp_path,FakeBackend()).run(q,output_dir=tmp_path/"runs",progress=lambda s:None)
    changed = q.copy(deep=True)
    changed.at[0,"reference_evidence"] = [{"claim_index":1,"accession":ACC,"supporting_passage":"unfindable xyz"}]
    backend = FakeBackend(); backend.forbid = True
    second = engine(tmp_path,backend).run(changed,output_dir=tmp_path/"runs",progress=lambda s:None)
    assert first["directory"] != second["directory"]
    assert not second["claims"].claim_complete.any()


def test_rerank_reuses_identical_pool_despite_order_and_provenance(tmp_path):
    backend = FakeBackend(); benchmark = engine(tmp_path,backend)
    cs = [chunk(0,"one"), chunk(1,"two")]
    first,_ = benchmark.rerank("question", cs)
    second,timing = benchmark.rerank("question", [{**c,"retrieval_provenance":[{"other":"strategy"}]} for c in reversed(cs)])
    assert backend.calls == ["rerank"]
    assert timing["cache_hit"]
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    benchmark.rerank("changed question", cs)
    assert backend.calls == ["rerank", "rerank"]


def test_question_settings_and_collection_change_invalidate_relevant_caches(tmp_path):
    base = engine(tmp_path, FakeBackend())
    strategy = rb.STRATEGIES[0]
    key = base.pool_key("q", strategy)
    assert key != base.pool_key("q changed", strategy)
    assert key != engine(tmp_path, FakeBackend(), settings={"embedding_model":"changed"}).pool_key("q",strategy)
    # Reranker settings do not invalidate candidate retrieval.
    assert key == engine(tmp_path, FakeBackend(), settings={"reranker_max_length":1000}).pool_key("q",strategy)
    base.identity = "changed-index"
    assert key != base.pool_key("q",strategy)
    cached = rb.DiskCache(tmp_path/"standalone")
    def fail():
        raise ValueError("failure")
    with pytest.raises(ValueError):
        cached.get("x",key,fail)
    assert not cached.path("x",key).exists()


def test_saved_legacy_scores_are_reproduced():
    directory = ROOT/"data/rag_evaluation/results/reranker_ablation/2026_09_15_222718"
    if not directory.exists():
        pytest.skip("Historical local result cache not present")
    questions = load_evaluation_questions(ROOT/"data/rag_evaluation/evaluation_questions_v3.csv").set_index("question_id")
    correct, count = 0,0
    for line in (directory/"experiment20_candidates_and_rankings.jsonl").read_text(encoding="utf-8").splitlines():
        saved = json.loads(line); q = questions.loc[saved["question_id"]]
        value = calculate_claim_evidence_recall(q.required_claims,q.reference_evidence,
                                               saved["metadata_ranked"][:20],allow_adjacent=False)
        correct += value*len(q.required_claims); count += len(q.required_claims)
    expected = pd.read_csv(directory/"reranker_summary.csv").set_index("stage").loc["metadata_20","claim_recall"]
    assert correct/count == pytest.approx(expected)


def test_fresh_ipython_kernel_run_all_with_mocks(tmp_path):
    """Execute actual notebook code in isolated kernels; all expensive backends are fake."""
    import sys
    from jupyter_client import KernelManager
    notebook = json.loads((ROOT/"src/RAG_model/model_analysis_notebooks/retrieval_methods.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"
        and 'experiment3_full_evaluation' not in c.get('metadata', {}).get('tags', []))
    for replay in (False, True):
        bootstrap = f'''
import sys, runpy
from pathlib import Path
sys.path.insert(0, {str(ROOT/'src')!r})
from RAG_model.retrieval import retrieval_benchmark as rb
FakeBackend = runpy.run_path({str(Path(__file__).resolve())!r})['FakeBackend']
real_benchmark = rb.Benchmark
fake_backend = FakeBackend()
fake_backend.forbid = {replay!r}
def mock_factory(root, **kwargs):
    obj = real_benchmark(root, backend=fake_backend, settings=kwargs.get('settings'),
                        cache_dir=Path({str(tmp_path/'cache')!r}), collection_identity='isolated-kernel')
    real_run = obj.run
    obj.run = lambda questions, **unused: real_run(questions, output_dir=Path({str(tmp_path/'runs')!r}), progress=lambda s:None)
    return obj
rb.Benchmark = mock_factory
'''
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        km.start_kernel(cwd=str(ROOT))
        client = km.blocking_client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
            message_id = client.execute(bootstrap + code + "\nassert len(result['questions']) == 256\nassert result['manifest']['status'] == 'complete'\n")
            while True:
                message = client.get_iopub_msg(timeout=60)
                if message.get("parent_header",{}).get("msg_id") != message_id:
                    continue
                if message["msg_type"] == "error":
                    pytest.fail("\n".join(message["content"]["traceback"]))
                if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                    break
        finally:
            client.stop_channels()
            km.shutdown_kernel(now=True)


def test_fresh_notebook_run_all_and_cache_only_replay(tmp_path, monkeypatch):
    import nbformat
    notebook = nbformat.read(ROOT/"src/RAG_model/model_analysis_notebooks/retrieval_methods.ipynb", as_version=4)
    nbformat.validate(notebook)
    real_class = rb.Benchmark
    backends = []
    def factory(root, **kwargs):
        backend = FakeBackend(); backend.forbid = bool(backends)
        backends.append(backend)
        instance = real_class(root,backend=backend,settings=kwargs.get("settings"),
                             cache_dir=tmp_path/"cache",collection_identity="test-index")
        real_run = instance.run
        def run(questions, **unused):
            return real_run(questions,output_dir=tmp_path/"runs",progress=lambda s:None)
        instance.run = run
        return instance
    monkeypatch.setattr(rb,"Benchmark",factory)
    monkeypatch.chdir(ROOT/"src/RAG_model/model_analysis_notebooks")
    for _ in range(2):
        namespace = {"__name__":"__main__"}
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'experiment3_full_evaluation' not in cell.metadata.get('tags', []):
                exec(compile(cell.source,"notebook-cell", "exec"), namespace)
        assert len(namespace["overall_results"]) == 8
        assert len(namespace["category_results"]) == 32
        assert len(namespace["result"]["questions"]) == 256
    assert backends[0].calls
    assert backends[1].calls == []
