"""No paid calls, real model inference, or mutations of the existing index."""
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from RAG_model.retrieval import full_benchmark as fb
from RAG_model.retrieval.evidence import load_evaluation_questions
from RAG_model.retrieval.result_metrics import score_answer, ABSTENTION

ROOT = Path(__file__).resolve().parents[1]
TAG = 'experiment3_full_evaluation'


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.forbid = set()
        self.fail_metric = None
        self.finish_reason = 'stop'
        self.empty_answer = False
        self.nan_metric = False
        self.closed = False

    def record(self, kind, value=None):
        if kind in self.forbid:
            raise AssertionError('Forbidden external computation: ' + kind)
        self.calls.append((kind, value))

    def catalog(self, collection):
        self.record('catalog')
        return [{'ticker': t, 'form_type': f, 'period_end': p,
                 'accession_number': f'000000000{n}-25-000001'}
                for n,t in enumerate(fb.SUPPORTED_TICKERS,1)
                for f,p in [('10-K','2024-12-31'),('10-K','2025-12-31'),('10-Q','2025-03-31')]]

    def embed(self, text):
        self.record('embed', text)
        return {'dense': [1.,0.], 'sparse': {'indices': [1], 'values': [1.]}}

    def search(self, collection, vector, spec, limit):
        self.record('search', {'spec': spec, 'limit': limit})
        ticker = spec['ticker'] or 'FETH'
        acc = f'000000000{fb.SUPPORTED_TICKERS.index(ticker)+1}-25-000001'
        scope = (spec['scopes'] or [{'form_type':'10-K','period_end':'2024-12-31'}])[0]
        return [{'chunk_id':f'{acc}*item-1*text-{i:04}', 'chunk_text':'alpha bravo' if i%2==0 else 'charlie delta',
                 'ticker':ticker, 'accession_number':acc, 'section_key':'item-1', 'section_title':'Business',
                 'source_url':'https://www.sec.gov/example', 'score':1., **scope} for i in range(limit)]

    def rerank(self, question, passages):
        self.record('rerank', len(passages))
        return [float(i) for i in range(len(passages))]

    async def generate(self, messages, settings):
        self.record('generate', messages)
        ids = re.findall(r'\[SOURCE: ([^\]]+)\]\n    Ticker:', messages[0]['content'])
        return {'answer': '' if self.empty_answer else ('alpha bravo charlie delta [SOURCE: ' + ids[0] + ']' if ids else ABSTENTION),
                'finish_reason':self.finish_reason, 'input_tokens':100, 'output_tokens':10, 'total_tokens':110}

    async def metric(self, name, sample):
        self.record('metric', name)
        if name == self.fail_metric:
            raise RuntimeError('Simulated metric interruption')
        return float('nan') if self.nan_metric else {'faithfulness':.9,'factual_correctness':.8,'context_precision':.7}[name]

    async def aclose(self):
        self.closed = True


def questions():
    return load_evaluation_questions(ROOT / 'data/rag_evaluation/evaluation_questions_v4.csv')


def runner(tmp_path, backend, **kw):
    return fb.FullBenchmark(ROOT, backend=backend, cache_dir=tmp_path/'full_cache',
        retrieval_cache_dir=tmp_path/'retrieval_cache', collection_identity='fake-index', **kw)


def run(tmp_path, backend, q=None, **kw):
    return asyncio.run(runner(tmp_path,backend,**kw).run(questions() if q is None else q,
        output_dir=tmp_path/'runs', progress=lambda s:None))


def test_text_scopes_all_unanswerable_and_mixed_periods():
    catalog = FakeBackend().catalog('fake')
    data = questions().set_index('question_id')
    resolved = {i:fb.resolve_searches(fb.question_constraints(data.loc[i,'question']), catalog) for i in range(33,41)}
    assert resolved[33][0]['empty']  # Q2 2028 is NOT the latest year.
    assert resolved[34][0]['empty'] and resolved[34][0]['spec']['ticker']=='COIN'
    assert resolved[35][0]['empty']  # before 2016, across every ticker
    assert not resolved[36][0]['empty'] and resolved[36][0]['spec']['ticker'] is None
    assert all(s['period_end'].startswith('2025') for s in resolved[36][0]['spec']['scopes'])
    assert resolved[37][0]['missing_periods']==['2023']
    assert resolved[37][0]['spec']['scopes']==[{'form_type':'10-K','period_end':'2025-12-31'}]
    for i in (38,39):
        assert resolved[i][0]['spec']==fb.filter_spec()
    assert resolved[40][0]['missing_periods']==['2020']
    assert resolved[40][0]['spec']['scopes']==[{'form_type':'10-K','period_end':'2024-12-31'}]
    mixed = fb.question_constraints(data.loc[19,'question'])['scopes']
    assert mixed['IBIT'][0]['period_end']=='2025-03-31'
    assert mixed['GBTC'][0]['period_end']=='2024-12-31'


def test_four_variants_all_40_and_cached_replay(tmp_path):
    backend = FakeBackend()
    result = run(tmp_path,backend)
    assert backend.closed
    assert result['manifest']['status']=='complete'
    assert len(result['questions'])==160 and len(result['overall'])==4 and len(result['category'])==20
    assert set(result['questions'].variant)=={v['variant'] for v in fb.VARIANTS}
    assert all(v['limit']<=20 for kind,v in backend.calls if kind=='search')
    frame = result['questions']
    for (_,cap), group in frame.groupby(['question_id','candidate_cap']):
        assert group.candidate_count.le(cap).all()
        for col in ('claim_evidence_recall_at_k','document_recall_at_k','document_hit_at_k'):
            assert group[col].dropna().nunique()<=1
    components = frame[['retrieval_latency_ms','reranking_latency_ms','prompt_latency_ms','generation_latency_ms']].sum(axis=1)
    assert frame.total_latency_ms.tolist()==pytest.approx(components.tolist())
    assert frame.loc[~frame.rerank,'reranking_latency_ms'].eq(0).all()
    assert frame.loc[~frame.answerable,'ragas_faithfulness'].isna().all()
    assert frame.loc[~frame.answerable,'claim_evidence_recall_at_k'].isna().all()
    assert frame.loc[~frame.answerable,'abstention_correct'].notna().all()
    assert result['overall'].answerable_questions.eq(32).all()
    assert result['overall'].unanswerable_questions.eq(8).all()
    assert result['overall'].mean_ragas_factual_correctness.eq(.8).all()
    assert result['overall'].category_min_mean_ragas_factual_correctness.eq(.8).all()
    assert result['category'].query("category == 'Unanswerable'").claim_recall.isna().all()
    for path in (result['directory']/'questions').glob('*.json'):
        saved = json.loads(path.read_text())
        assert {c['chunk_id'] for c in saved['context']}=={c['chunk_id'] for c in saved['pool']['candidates']}
    again = FakeBackend(); again.forbid={'embed','search','catalog','rerank','generate','metric'}
    replay = run(tmp_path,again)
    assert not again.calls and replay['directory']==result['directory']
    preflight = runner(tmp_path,again).preflight(questions().to_dict('records'))
    for counts in preflight['stages'].values():
        assert counts['missing']==0 and counts['inputs_pending']==0
    assert preflight['stages']['generation']['cached']==160
    assert preflight['stages']['ragas_metric']['cached']==384
    assert replay['questions'].latency_provenance.eq('reconstructed_from_recorded_stages').all()
    assert replay['questions'].generation_latency_ms.tolist()==frame.generation_latency_ms.tolist()
    assert replay['questions'].generation_total_tokens.eq(110).all()


def test_smaller_multi_ticker_depth_and_gold_never_enters_retrieval(tmp_path):
    backend = FakeBackend(); bench = runner(tmp_path,backend)
    q = questions().set_index('question_id').loc[19].question
    pool,_ = bench.pool(q,15)
    assert pool['per_ticker_depth']==8 and len(pool['candidates'])==15
    assert [v['limit'] for k,v in backend.calls if k=='search']==[8,8]
    assert all(v==q for k,v in backend.calls if k=='embed')
    for k,v in backend.calls:
        if k=='search':
            t = v['spec']['ticker']
            assert v['spec']['scopes'][0]['period_end']==('2025-03-31' if t=='IBIT' else '2024-12-31')


def test_metric_failure_resumes_and_history_survives(tmp_path):
    first = FakeBackend(); first.fail_metric='factual_correctness'
    q = questions().head(1)
    with pytest.raises(RuntimeError, match='Simulated'):
        run(tmp_path,first,q)
    assert first.closed
    failure = next((tmp_path/'runs').glob('*/failures/*.json'))
    assert json.loads(failure.read_text())['stage']=='ragas_factual_correctness'
    assert len(list((tmp_path/'full_cache'/'generation').glob('*.json')))==1
    assert len(list((tmp_path/'full_cache'/'ragas_faithfulness').glob('*.json')))==1
    assert not list((tmp_path/'full_cache'/'ragas_factual_correctness').glob('*.json'))
    second = FakeBackend()
    finished = run(tmp_path,second,q)
    assert failure.exists() and finished['manifest']['status']=='complete'
    assert second.calls[0]==('metric','factual_correctness')
    assert len(list((finished['directory']/'attempts').glob('*.json')))==2


@pytest.mark.parametrize('bad', ['length','empty','nan'])
def test_invalid_generation_or_metric_is_not_successfully_cached(tmp_path,bad):
    backend = FakeBackend()
    if bad=='length': backend.finish_reason='length'
    if bad=='empty': backend.empty_answer=True
    if bad=='nan': backend.nan_metric=True
    with pytest.raises(ValueError):
        run(tmp_path,backend,questions().head(1))
    directory = next((tmp_path/'runs').iterdir())
    assert json.loads((directory/'manifest.json').read_text())['status']=='incomplete'
    failure = json.loads(next((directory/'failures').glob('*.json')).read_text())
    if bad!='nan':
        assert 'response' in failure
        assert not list((tmp_path/'full_cache'/'generation').glob('*.json'))
    else:
        assert not list((tmp_path/'full_cache'/'ragas_faithfulness').glob('*.json'))


def test_length_generation_retries_once_with_expanded_limit_and_caches(tmp_path):
    class ExpandingBackend(FakeBackend):
        async def generate(self, messages, settings):
            limit = settings['generation_max_tokens']
            self.record('generate_limit', limit)
            ids = re.findall(r'\[SOURCE: ([^\]]+)\]\n    Ticker:', messages[0]['content'])
            return {
                'answer': ('partial' if limit == 1000 else
                           ('alpha bravo [SOURCE: ' + ids[0] + ']' if ids else ABSTENTION)),
                'finish_reason': 'length' if limit == 1000 else 'stop',
                'input_tokens': 100,
                'output_tokens': limit,
                'total_tokens': 100 + limit,
            }

    backend = ExpandingBackend()
    result = run(tmp_path, backend, questions().head(1))
    limits = [value for kind, value in backend.calls if kind == 'generate_limit']
    assert limits == [1000, 2000] * 4
    assert result['questions'].generation_output_tokens.eq(2000).all()

    replay_backend = ExpandingBackend()
    replay_backend.forbid = {'generate_limit'}
    replay = run(tmp_path, replay_backend, questions().head(1))
    assert replay['questions'].generation_cache_hit.all()
    assert not replay_backend.calls


def test_compatible_resume_forces_known_length_key_directly_to_retry_limit(tmp_path):
    class ExpandingBackend(FakeBackend):
        async def generate(self, messages, settings):
            limit = settings['generation_max_tokens']
            self.record('generate_limit', limit)
            return {'answer': 'complete', 'finish_reason': 'stop', 'input_tokens': 1,
                    'output_tokens': 2, 'total_tokens': 3}

    backend = ExpandingBackend()
    identity = 'a' * 64
    bench = runner(tmp_path, backend, implementation_identity=identity)
    messages, key = bench.messages_and_key('question', [])
    assert key['implementation'] == identity
    assert 'retry_max_tokens' not in key['generation']
    bench.force_retry_generation_keys.add(fb.digest(key))
    result, _ = asyncio.run(bench.generation(messages, key))
    assert [value for kind, value in backend.calls if kind == 'generate_limit'] == [2000]
    assert result['effective_max_tokens'] == 2000
    assert result['generation_attempts'] == [
        {'max_tokens': 2000, 'finish_reason': 'stop', 'output_tokens': 2}
    ]


def test_reference_change_reuses_retrieval_and_generation(tmp_path):
    q = questions().head(1)
    first = run(tmp_path,FakeBackend(),q)
    q = q.copy(deep=True)
    q.loc[q.index[0],'reference_answer']='Changed reference, evaluation only'
    backend = FakeBackend(); backend.forbid={'embed','search','catalog','rerank','generate'}
    second = run(tmp_path,backend,q)
    assert first['directory']!=second['directory'] and all(k=='metric' for k,_ in backend.calls)
    assert second['questions'].generation_cache_hit.all()


def test_cache_keys_and_prompt_content(tmp_path):
    backend = FakeBackend(); bench = runner(tmp_path,backend)
    text = questions().iloc[0].question
    p,_ = bench.pool(text,15)
    messages,key = bench.messages_and_key(text,p['candidates'])
    assert messages[-1]['content']==text
    assert 'reference_answer' not in str(messages) and 'required_claims' not in str(messages)
    assert key != bench.messages_and_key(text,list(reversed(p['candidates'])))[1]
    changed = runner(tmp_path,backend,settings={'generation_model':'different'})
    assert key != changed.messages_and_key(text,p['candidates'])[1]
    assert bench.pool_key(text,15)==changed.pool_key(text,15)
    changed = runner(tmp_path,backend,settings={'embedding_model':'different'})
    assert bench.pool_key(text,15)!=changed.pool_key(text,15)
    bench.retrieval.identity='different-index'
    assert bench.pool_key(text,15)!=runner(tmp_path,backend).pool_key(text,15)


def test_deterministic_citations_abstention_and_claim_denominators():
    q = {'answerable':True,'expected_document':'a;b', 'required_claims':['one','two'],
         'reference_evidence':[{'claim_index':1,'accession':'123','supporting_passage':'alpha'}]}
    cs = [{'chunk_id':'x','accession_number':'123','chunk_text':'alpha'}]
    scored = score_answer(q,cs,'alpha [SOURCE: invented]')
    assert scored['claim_evidence_recall_at_k']==.5
    assert not scored['citations_valid'] and scored['invalid_citations']==['invented']
    assert not score_answer(q,cs,'alpha')['citations_valid']
    assert score_answer(q,cs,ABSTENTION)['incorrect_abstention']
    q['answerable']=False
    assert score_answer(q,[],ABSTENTION)['abstention_correct']
    assert not score_answer(q,[],ABSTENTION+' extra')['abstention_correct']


def test_identical_empty_contexts_reuse_generation_without_skipping_abstention(tmp_path):
    q = questions().query('question_id == 33')
    backend = FakeBackend()
    result = run(tmp_path,backend,q)
    assert len(result['questions'])==4
    assert result['questions'].context_count.eq(0).all()
    assert result['questions'].abstention_correct.all()
    assert [k for k,_ in backend.calls].count('generate')==1
    assert not any(k in ('embed','search','rerank','metric') for k,_ in backend.calls)


def test_scope_result_filter_violations_fail_loudly(tmp_path):
    class WrongTicker(FakeBackend):
        def search(self,*args):
            chunks = super().search(*args)
            for chunk in chunks:
                chunk['ticker']='WRONG'
            return chunks
    with pytest.raises(AssertionError,match='Filter violation'):
        runner(tmp_path,WrongTicker()).pool(questions().iloc[0].question,15)


def test_metadata_reranker_receives_full_question_and_passages(tmp_path):
    class InspectBackend(FakeBackend):
        def rerank(self,question,passages):
            assert 'Source scope:' in question
            assert all('Ticker:' in p and 'Reporting period:' in p and 'Content:' in p for p in passages)
            return super().rerank(question,passages)
    run(tmp_path,InspectBackend(),questions().head(1))


def test_weighted_claims_and_unanswerable_aggregation_are_separate(tmp_path):
    data = questions().iloc[[0,1,32]].copy(deep=True)
    for idx,claims in zip(data.index[:2], [['one'], ['a','b','c']]):
        ticker = next(iter(fb.source_scope(data.at[idx,'question'])))
        accession = f'000000000{fb.SUPPORTED_TICKERS.index(ticker)+1}-25-000001'
        data.at[idx,'required_claims']=claims
        data.at[idx,'reference_evidence']=[{'claim_index':1,'accession':accession,
            'supporting_passage':'alpha bravo charlie delta'}]
    result = run(tmp_path,FakeBackend(),data)
    # One supported claim per question / four total (not a mean of question recalls).
    assert result['overall'].claim_recall.eq(.5).all()
    assert result['overall'].unanswerable_questions.eq(1).all()
    assert result['overall'].abstention_accuracy.eq(1).all()


def test_underfilled_pools_are_reported_without_backfill(tmp_path):
    class ShortBackend(FakeBackend):
        def search(self,*args):
            return super().search(*args)[:2]
    result = run(tmp_path,ShortBackend(),questions().head(1))
    assert result['questions'].candidate_count.eq(2).all()
    assert result['questions'].context_count.eq(2).all()
    assert result['overall'].underfilled_questions.eq(1).all()


def test_ragas_factory_preserves_config_without_network(monkeypatch):
    import openai
    import ragas.llms
    import ragas.metrics
    seen = {}
    monkeypatch.setenv('OPENROUTER_API_KEY','test-only-not-a-key')
    monkeypatch.setattr(openai,'AsyncOpenAI',lambda **kw: object())
    monkeypatch.setattr(ragas.llms,'llm_factory',lambda *a,**kw: object())
    for cls in ('Faithfulness','FactualCorrectness','LLMContextPrecisionWithReference'):
        def capture(*, _name=cls, **kwargs):
            seen[_name]=kwargs
            return object()
        monkeypatch.setattr(ragas.metrics,cls,capture)
    fb.ragas_factory.create_ragas_metrics({'ragas_evaluator_model':'mock-model','ragas_temperature':0})
    assert seen['FactualCorrectness']['mode']=='recall'
    assert seen['FactualCorrectness']['atomicity']=='high'
    assert seen['FactualCorrectness']['coverage']=='high'


def test_fresh_kernel_only_new_section_then_cache_replay(tmp_path):
    import nbformat
    from jupyter_client import KernelManager
    notebook = nbformat.read(ROOT/'src/RAG_model/model_analysis_notebooks/retrieval_methods.ipynb',as_version=4)
    nbformat.validate(notebook)
    cells = [c.source for c in notebook.cells if c.cell_type=='code' and TAG in c.metadata.get('tags',[])]
    assert len(cells)==7
    for replay in (False,True):
        bootstrap = f'''
import sys, runpy
from pathlib import Path
sys.path.insert(0, {str(ROOT/'src')!r})
from RAG_model.retrieval import full_benchmark as fb
FakeBackend = runpy.run_path({str(Path(__file__).resolve())!r})['FakeBackend']
original = fb.FullBenchmark
fake = FakeBackend()
if {replay!r}:
    fake.forbid = {{'embed','search','catalog','rerank','generate','metric'}}
def factory(root, **kwargs):
    obj = original(root, settings=kwargs.get('settings'), backend=fake,
        cache_dir=Path({str(tmp_path/'cache')!r}), retrieval_cache_dir=Path({str(tmp_path/'retrieval')!r}),
        collection_identity='kernel-test')
    real_run = obj.run
    async def run(q, **ignored):
        return await real_run(q, output_dir=Path({str(tmp_path/'runs')!r}), progress=lambda s:None)
    obj.run = run
    return obj
fb.FullBenchmark = factory
'''
        km = KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        km.start_kernel(cwd=str(ROOT/'src/RAG_model/model_analysis_notebooks'))
        client = km.blocking_client(); client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
            for source in [bootstrap,*cells,"assert len(full_result['questions'])==160\nassert len(full_category)==20\nassert full_result['manifest']['status']=='complete'"]:
                mid = client.execute(source)
                while True:
                    msg = client.get_iopub_msg(timeout=60)
                    if msg.get('parent_header',{}).get('msg_id')!=mid: continue
                    if msg['msg_type']=='error': pytest.fail('\n'.join(msg['content']['traceback']))
                    if msg['msg_type']=='status' and msg['content']['execution_state']=='idle': break
        finally:
            client.stop_channels(); km.shutdown_kernel(now=True)
