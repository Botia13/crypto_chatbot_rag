"""Independent, resumable experiment-3 generation benchmark. Imports are inert.

Retrieval accepts question text only. Gold fields enter deterministic/RAGAS scoring
after generation. Existing retrieval caches and historical results are read/reused,
never rewritten by changing the original benchmark's implementation fingerprint.
"""
import asyncio
import calendar
import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from RAG_model.answer.prompt import make_rag_messages
from RAG_model.retrieval import result_metrics
from RAG_model.retrieval.result_metrics import is_answerable, score_answer
from RAG_model.retrieval.retrieval_benchmark import (
    Benchmark, LocalBackend, DiskCache, DEFAULT_SETTINGS, COLLECTION, atomic_json,
    digest, source_scope, filter_spec, assert_scope, round_robin, deduplicate, metadata_passage,
)
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, SYSTEM_PROMPT
from RAG_model.retrieval import evidence, ragas_factory

VARIANTS = tuple({'variant': f'direct_{k}_{"reranked" if rank else "raw"}',
                  'candidate_cap': k, 'rerank': rank}
                 for k in (15, 20) for rank in (False, True))
RAGAS_METRICS = ('faithfulness', 'factual_correctness', 'context_precision')
ANSWERABLE_CATEGORIES = ('Factual', 'Procedural', 'Comparison', 'Synthesis')
SUPPORTED_TICKERS = ('IBIT', 'ETHA', 'FBTC', 'FETH', 'GBTC', 'ETHE')
METRIC_SETTINGS = {'factual_correctness_mode': 'recall', 'atomicity': 'high', 'coverage': 'high'}
DEFAULT_GENERATION_RETRY_MAX_TOKENS = 2000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def question_constraints(question):
    """Resolve only explicit text. Unknown products never map to a known fund."""
    if 'Source scope:' in question:
        return {'mode': 'explicit_scope', 'scopes': source_scope(question)}
    tickers = re.findall(r'\b(?:' + '|'.join(SUPPORTED_TICKERS) + r')\b', question.upper())
    tickers += [t.upper().rstrip('.-') for t in re.findall(r'\bticker\s+([A-Za-z][A-Za-z0-9.-]*)', question, re.I)]
    tickers = list(dict.fromkeys(tickers))
    forms = re.findall(r'\b10-[KQ]\b', question.upper())
    if re.search(r'\bannual\b', question, re.I):
        forms.append('10-K')
    before = re.search(r'\bbefore\s+((?:19|20)\d{2})\b', question, re.I)
    quarters = re.findall(r'\bQ([1-4])\s+(?:of\s+)?((?:19|20)\d{2})\b', question, re.I)
    dates = [f'{int(y):04d}-{int(q)*3:02d}-{calendar.monthrange(int(y), int(q)*3)[1]:02d}' for q,y in quarters]
    years = sorted(set(re.findall(r'\b(?:19|20)\d{2}\b', question)))
    # Quarter/before years are not also expanded into an entire year.
    years = [y for y in years if y not in {y for _,y in quarters} and not (before and y == before[1])]
    return {'mode': 'text_constraints', 'tickers': tickers, 'forms': sorted(set(forms)),
            'period_dates': dates, 'period_years': years,
            'before': f'{before[1]}-01-01' if before else None,
            'unresolved_entity': not tickers,
            'unrestricted_period': not (dates or years or before)}


def resolve_searches(constraints, catalog):
    """Return one entry per ticker, including explicit empty entries for no match.

Catalog contains only public filing metadata from the indexed collection, not gold.
The entry count still controls depth allocation when one ticker has no matches.
"""
    if constraints['mode'] == 'explicit_scope':
        return [{'spec': filter_spec(t, scopes), 'empty': False, 'missing_periods': []}
                for t,scopes in constraints['scopes'].items()]
    c = constraints
    entries = []
    for ticker in c['tickers'] or [None]:
        available = [f for f in catalog if (not ticker or f['ticker'] == ticker)
                     and (not c['forms'] or f['form_type'] in c['forms'])]
        def period_matches(f):
            period = f['period_end']
            if c['unrestricted_period']:
                return True
            return (period in c['period_dates'] or period[:4] in c['period_years']
                    or bool(c['before'] and period < c['before']))
        matches = [f for f in available if period_matches(f)]
        requested = c['period_dates'] + c['period_years']
        missing = [p for p in requested if not any(f['period_end'].startswith(p) for f in matches)]
        constrained = bool(c['forms'] or not c['unrestricted_period'])
        scopes = sorted({(f['form_type'], f['period_end']) for f in matches}) if constrained else []
        entries.append({'spec': filter_spec(ticker, [{'form_type': f, 'period_end': p} for f,p in scopes]),
                        'empty': not matches, 'missing_periods': missing,
                        'matching_filings': len(matches)})
    return entries


class FullBackend(LocalBackend):
    def __init__(self, root, settings):
        super().__init__(root, settings)
        self._ragas_client = self._metrics = None

    def catalog(self, collection):
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(path=str(self.root / 'data/qdrant_storage'))
        fields = ['ticker', 'form_type', 'period_end', 'accession_number']
        filings, offset = {}, None
        while True:
            points, offset = self._qdrant.scroll(collection, limit=256, offset=offset,
                with_payload=fields, with_vectors=False)
            for point in points:
                p = point.payload or {}
                if not all(p.get(f) for f in fields):
                    raise ValueError('Indexed filing missing scope metadata')
                item = {f: p[f] for f in fields}
                previous = filings.setdefault(p['accession_number'], item)
                if previous != item:
                    raise ValueError('Conflicting scope metadata within an accession')
            if offset is None:
                break
        return list(filings.values())

    async def generate(self, messages, settings):
        response = await asyncio.to_thread(self.api.chat.completions.create,
            model=settings['generation_model'], temperature=settings['temperature'],
            max_tokens=settings['generation_max_tokens'], messages=messages)
        choice = response.choices[0] if response.choices else None
        usage = response.usage
        return {'answer': choice.message.content if choice and choice.message else '',
                'finish_reason': choice.finish_reason if choice else None,
                'response_id': response.id, 'response_model': response.model,
                'input_tokens': usage.prompt_tokens if usage else None,
                'output_tokens': usage.completion_tokens if usage else None,
                'total_tokens': usage.total_tokens if usage else None}

    async def metric(self, name, sample):
        from ragas import SingleTurnSample
        if self._metrics is None:
            from dotenv import load_dotenv
            load_dotenv(self.root / '.env')
            self._ragas_client, self._metrics = ragas_factory.create_ragas_metrics(self.settings)
        return float(await self._metrics[name + '_metric'].single_turn_ascore(SingleTurnSample(**sample)))

    async def aclose(self):
        super().close()
        if self._api is not None:
            self._api.close()
        if self._ragas_client is not None:
            await self._ragas_client.close()


class AsyncCache(DiskCache):
    async def aget(self, stage, key, compute):
        if self.path(stage, key).exists():
            return self.get(stage, key, lambda: None)
        started = time.perf_counter()
        value = await compute()
        duration = time.perf_counter() - started
        atomic_json(self.path(stage, key), {'key_digest': digest(key), 'value': value,
                    'compute_seconds': duration, 'created_at': utc_now()})
        elapsed = time.perf_counter() - started
        self.events.append({'stage': stage, 'hit': False, 'elapsed_seconds': elapsed})
        return value, {'cache_hit': False, 'wall_seconds': elapsed, 'saved_compute_seconds': duration}


class GenerationFailure(ValueError):
    def __init__(self, response, attempts=None):
        super().__init__(f"Empty or unsuccessful generation (finish_reason={response.get('finish_reason')})")
        self.response = response
        self.attempts = attempts or []


class FullBenchmark:
    def __init__(self, root, *, settings=None, backend=None, cache_dir=None,
                 retrieval_cache_dir=None, collection=COLLECTION, collection_identity=None,
                 system_prompt=SYSTEM_PROMPT, implementation_identity=None,
                 generation_retry_max_tokens=DEFAULT_GENERATION_RETRY_MAX_TOKENS):
        self.root = Path(root)
        self.settings = {**DEFAULT_SETTINGS, **{k: BASELINE_RUN_CONFIG[k] for k in (
            'generation_model', 'temperature', 'prompt_version', 'ragas_evaluator_model',
            'ragas_temperature', 'answer_correct_threshold')}, 'generation_max_tokens': 1000,
            **(settings or {})}
        self.system_prompt = system_prompt
        self.backend = backend or FullBackend(root, self.settings)
        self.cache = AsyncCache(cache_dir or self.root / 'data/rag_evaluation/results/experiment3_full_evaluation/cache')
        retrieval_settings = {k: self.settings[k] for k in DEFAULT_SETTINGS}
        self.retrieval = Benchmark(root, settings=retrieval_settings, backend=self.backend,
            cache_dir=retrieval_cache_dir, collection=collection, collection_identity=collection_identity)
        computed_code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if implementation_identity is not None and not re.fullmatch(r'[0-9a-f]{64}', implementation_identity):
            raise ValueError('implementation_identity must be a 64-character lowercase SHA-256 digest')
        self.code = implementation_identity or computed_code
        self.resuming_compatible_implementation = implementation_identity is not None
        self.generation_retry_max_tokens = int(generation_retry_max_tokens)
        if self.generation_retry_max_tokens < self.settings['generation_max_tokens']:
            raise ValueError('generation_retry_max_tokens cannot be smaller than generation_max_tokens')
        self.force_retry_generation_keys = set()
        self.metric_code = digest([fingerprint(evidence), fingerprint(result_metrics)])
        self.ragas_code = digest([fingerprint(ragas_factory), version('ragas'), METRIC_SETTINGS])

    def pool_key(self, question, cap):
        return {'implementation': self.code, 'question': question, 'cap': cap,
                'collection': self.retrieval.identity, 'retrieval': self.retrieval.implementation,
                'embedding_model': self.settings['embedding_model']}

    def pool(self, question, cap):
        def compute():
            constraints = question_constraints(question)
            catalog = []
            if constraints['mode'] != 'explicit_scope':
                catalog, _ = self.cache.get('catalog', {'collection': self.retrieval.identity, 'code': self.code},
                    lambda: self.backend.catalog(self.retrieval.collection))
            entries = resolve_searches(constraints, catalog)
            depth = math.ceil(cap / len(entries))
            events_start = len(self.retrieval.cache.events)
            groups = [[] if entry['empty'] else self.retrieval.search(question, entry['spec'], depth,
                {'purpose': 'candidate', 'facet_id': 'full_question', 'query_kind': 'full_question'})
                for entry in entries]
            chunks = deduplicate(round_robin(groups), cap)
            return {'candidates': chunks, 'scope': constraints, 'searches': entries,
                    'per_ticker_depth': depth,
                    'intermediate_cache_hits': sum(e['hit'] for e in self.retrieval.cache.events[events_start:])}
        pool, timing = self.cache.get('pool', self.pool_key(question, cap), compute)
        for c in pool['candidates']:
            for provenance in c.get('retrieval_provenance', []):
                assert_scope([c], provenance['filter'])
        if len(pool['candidates']) > cap or len({c['chunk_id'] for c in pool['candidates']}) != len(pool['candidates']):
            raise AssertionError('Invalid candidate cap or duplicate chunk IDs')
        return pool, timing

    def messages_and_key(self, question, chunks):
        messages = make_rag_messages(question, [], chunks, self.system_prompt)
        key = {'messages': messages, 'generation': {k: self.settings[k] for k in
            ('generation_model', 'temperature', 'generation_max_tokens')}, 'implementation': self.code}
        # A compatibility resume deliberately preserves historical successful cache keys.
        # Fresh runs fingerprint the retry policy as part of their generation contract.
        if not self.resuming_compatible_implementation:
            key['generation']['retry_max_tokens'] = self.generation_retry_max_tokens
        return messages, key

    async def generation(self, messages, key):
        async def compute():
            base_limit = self.settings['generation_max_tokens']
            force_retry = digest(key) in self.force_retry_generation_keys
            first_limit = self.generation_retry_max_tokens if force_retry else base_limit
            first_settings = {**self.settings, 'generation_max_tokens': first_limit}
            result = await self.backend.generate(messages, first_settings)
            attempts = [{'max_tokens': first_limit, 'finish_reason': result.get('finish_reason'),
                         'output_tokens': result.get('output_tokens')}]
            if (result.get('finish_reason') == 'length'
                    and first_limit < self.generation_retry_max_tokens):
                retry_settings = {**self.settings,
                                  'generation_max_tokens': self.generation_retry_max_tokens}
                result = await self.backend.generate(messages, retry_settings)
                attempts.append({'max_tokens': self.generation_retry_max_tokens,
                                 'finish_reason': result.get('finish_reason'),
                                 'output_tokens': result.get('output_tokens')})
            if not str(result.get('answer') or '').strip() or result.get('finish_reason') != 'stop':
                raise GenerationFailure(result, attempts)
            result = {**result, 'generation_attempts': attempts,
                      'effective_max_tokens': attempts[-1]['max_tokens']}
            return result
        return await self.cache.aget('generation', key, compute)

    def metric_key(self, name, sample):
        # Context precision does not depend on the generated answer.
        inputs = {k:v for k,v in sample.items() if k != 'response'} if name == 'context_precision' else sample
        key = {'name': name, 'sample': inputs, 'model': self.settings['ragas_evaluator_model'],
               'temperature': self.settings['ragas_temperature'], 'metric_code': self.ragas_code,
               'implementation': self.code}
        return inputs, key

    async def judge(self, name, sample):
        inputs, key = self.metric_key(name, sample)
        async def compute():
            value = await self.backend.metric(name, inputs)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid {name} result: {value}')
            return value
        return await self.cache.aget('ragas_' + name, key, compute)

    def preflight(self, records):
        """Inspect cache records only; unknown downstream inputs never trigger work."""
        counts = {s: {'cached': 0, 'missing': 0, 'inputs_pending': 0}
                  for s in ('pool', 'reranking', 'generation', 'ragas_metric')}
        def read(cache, stage, key):
            path = cache.path(stage, key)
            if not path.exists():
                return None
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['key_digest'] != digest(key):
                raise ValueError(f'Invalid cache key in {path}')
            return record['value']
        for q in records:
            for cap in (15,20):
                pool = read(self.cache, 'pool', self.pool_key(q['question'],cap))
                counts['pool']['cached' if pool is not None else 'missing'] += 1
                for rank in (False,True):
                    chunks = pool['candidates'] if pool is not None else None
                    if rank:
                        if chunks is None:
                            counts['reranking']['inputs_pending'] += 1
                        elif chunks:
                            ordered = sorted(chunks, key=lambda c:c['chunk_id'])
                            rk = self.retrieval.key(question=q['question'],
                                passages=[(c['chunk_id'],metadata_passage(c)) for c in ordered],
                                model=self.settings['reranker_model'], max_length=self.settings['reranker_max_length'],
                                batch_size=self.settings['reranker_batch_size'])
                            scores = read(self.retrieval.cache,'rerank',rk)
                            counts['reranking']['cached' if scores is not None else 'missing'] += 1
                            chunks = sorted(chunks,key=lambda c:(-scores[c['chunk_id']],c['chunk_id'])) if scores is not None else None
                    generated = None
                    if chunks is None:
                        counts['generation']['inputs_pending'] += 1
                    else:
                        _, key = self.messages_and_key(q['question'],chunks)
                        generated = read(self.cache,'generation',key)
                        counts['generation']['cached' if generated is not None else 'missing'] += 1
                    if not is_answerable(q['answerable']):
                        continue
                    for name in RAGAS_METRICS:
                        if chunks is None or (generated is None and name != 'context_precision'):
                            counts['ragas_metric']['inputs_pending'] += 1
                            continue
                        sample = {'user_input':q['question'], 'reference':q['reference_answer'],
                            'response': generated['answer'] if generated else '',
                            'retrieved_contexts':[c['chunk_text'] for c in chunks]}
                        _,key = self.metric_key(name,sample)
                        value = read(self.cache,'ragas_'+name,key)
                        counts['ragas_metric']['cached' if value is not None else 'missing'] += 1
        return {'questions':len(records), 'variant_results':len(records)*4, 'stages':counts,
                'note':'Counts are per requested evaluation before identical-input deduplication; empty pools need no BGE.'}

    async def run(self, questions, *, output_dir=None, progress=print):
        records = questions.to_dict('records')
        if not records or len({q['question_id'] for q in records}) != len(records):
            raise ValueError('Questions must be nonempty with unique IDs')
        for q in records:
            if is_answerable(q['answerable']) and not q.get('required_claims'):
                raise ValueError('Answerable question missing required claims')
        signature = {'dataset': digest(records), 'settings': self.settings, 'system_prompt': self.system_prompt,
                     'collection': self.retrieval.identity, 'implementation': self.code,
                     'retrieval_implementation': self.retrieval.implementation,
                     'metric_code': self.metric_code, 'ragas_code': self.ragas_code, 'variants': VARIANTS}
        if not self.resuming_compatible_implementation:
            signature['generation_retry_max_tokens'] = self.generation_retry_max_tokens
        directory = Path(output_dir or self.root / 'data/rag_evaluation/results/experiment3_full_evaluation/runs') / digest(signature)[:20]
        directory.mkdir(parents=True, exist_ok=True)
        attempt_id = uuid.uuid4().hex
        manifest = {**signature, 'status': 'running', 'attempt_id': attempt_id,
                    'started_at': utc_now(), 'completed_results': 0, 'expected_results': len(records)*4}
        atomic_json(directory / 'manifest.json', manifest)
        rows = []
        context = {}
        try:
            progress('Independent experiment 3: ' + json.dumps(self.preflight(records)))
            progress('Inputs pending means upstream work is needed to determine the exact cache key. No old cells required.')
            for q in records:
                for cap in (15,20):
                    context = {'question_id': q['question_id'], 'candidate_cap': cap, 'stage': 'retrieval'}
                    retrieval_start = time.perf_counter()
                    own_event_start = len(self.cache.events)
                    search_event_start = len(self.retrieval.cache.events)
                    pool, rt = self.pool(q['question'], cap)
                    pool_wall = time.perf_counter() - retrieval_start
                    retrieval_cache_ms = 1000 * sum(e['elapsed_seconds'] for e in
                        self.cache.events[own_event_start:] + self.retrieval.cache.events[search_event_start:] if e['hit'])
                    candidates = pool['candidates']
                    invariant = None
                    for rerank in (False,True):
                        variant = f'direct_{cap}_{"reranked" if rerank else "raw"}'
                        context = {'question_id': q['question_id'], 'candidate_cap': cap,
                                   'variant': variant, 'stage': 'rerank' if rerank else 'generation'}
                        progress(f"Q{q['question_id']} {variant}: {len(candidates)} candidates; checking stage caches")
                        started = time.perf_counter()
                        stage_timings = {'retrieval': rt}
                        if rerank and candidates:
                            chunks, rr = self.retrieval.rerank(q['question'], candidates)
                        else:
                            chunks, rr = candidates, {'cache_hit': False, 'wall_seconds': 0., 'saved_compute_seconds': 0.}
                        stage_timings['reranking'] = rr
                        if sorted(c['chunk_id'] for c in chunks) != sorted(c['chunk_id'] for c in candidates):
                            raise AssertionError('Reranking changed evidence membership')
                        prompt_start = time.perf_counter()
                        messages, generation_key = self.messages_and_key(q['question'], chunks)
                        prompt_ms = (time.perf_counter() - prompt_start)*1000
                        context['stage'] = 'generation'
                        generated, gt = await self.generation(messages, generation_key)
                        stage_timings['generation'] = gt
                        evaluation_start = time.perf_counter()
                        deterministic, dt = self.cache.get('deterministic',
                            {'question': q, 'chunks': chunks, 'answer': generated['answer'], 'metric_code': self.metric_code},
                            lambda: score_answer(q, chunks, generated['answer']))
                        current = tuple(deterministic[k] for k in ('document_hit_at_k', 'document_recall_at_k',
                                        'claim_evidence_recall_at_k', 'legacy_claim_evidence_recall_at_k'))
                        if invariant is not None and invariant != current:
                            raise AssertionError('Order-invariant retrieval scores changed after reranking')
                        invariant = current
                        metric_values, metric_timings = {}, {}
                        sample = {'user_input': q['question'], 'response': generated['answer'],
                                  'reference': q['reference_answer'],
                                  'retrieved_contexts': [c['chunk_text'] for c in chunks]}
                        for name in RAGAS_METRICS:
                            context['stage'] = 'ragas_' + name
                            if is_answerable(q['answerable']):
                                metric_values[name], metric_timings[name] = await self.judge(name, sample)
                            else:
                                metric_values[name] = None
                        evaluation_wall_ms = (time.perf_counter() - evaluation_start)*1000
                        answerable = is_answerable(q['answerable'])
                        correct = (metric_values['factual_correctness'] >= self.settings['answer_correct_threshold']
                                   if answerable else deterministic['abstention_correct'])
                        raw_timing = {'retrieval_latency_ms': rt['saved_compute_seconds']*1000,
                                      'reranking_latency_ms': rr['saved_compute_seconds']*1000,
                                      'prompt_latency_ms': prompt_ms, 'generation_latency_ms': gt['saved_compute_seconds']*1000}
                        reused = rerank or pool['intermediate_cache_hits'] or any(t['cache_hit'] for t in stage_timings.values())
                        cache_ms = retrieval_cache_ms + 1000*sum(t['wall_seconds'] for t in
                            [rr,gt,dt,*metric_timings.values()] if t['cache_hit'])
                        row = {'question_id': q['question_id'], 'category': q['category'], 'answerable': answerable,
                               'variant': variant, 'candidate_cap': cap, 'rerank': rerank,
                               'candidate_count': len(candidates), 'context_count': len(chunks),
                               'underfilled': len(chunks)<cap, 'empty_pool': not chunks,
                               **{k:v for k,v in deterministic.items() if k not in ('claims','evidence','legacy_claims','legacy_evidence')},
                               **{'ragas_' + k: v for k,v in metric_values.items()}, 'answer_correct': correct,
                               **raw_timing, 'total_latency_ms': sum(raw_timing.values()),
                               'latency_provenance': 'reconstructed_from_recorded_stages' if reused else 'measured_stage_sum',
                               'evaluation_latency_ms': evaluation_wall_ms,
                               'evaluation_saved_ms': 1000*(dt['saved_compute_seconds'] + sum(t['saved_compute_seconds'] for t in metric_timings.values())),
                               'cache_load_ms': cache_ms,
                               'wall_ms_this_execution': (time.perf_counter()-started + (0 if rerank else pool_wall))*1000,
                               **{'generation_' + k: generated.get(k) for k in ('input_tokens','output_tokens','total_tokens')},
                               'embedding_tokens': None, 'rag_total_tokens': None,
                               'generation_cache_hit': gt['cache_hit'], 'retrieval_cache_hit': rt['cache_hit'],
                               'reranker_cache_hit': rr['cache_hit'], 'answer': generated['answer']}
                        checkpoint = {'result': row, 'question': q, 'pool': pool, 'context': chunks,
                                      'messages': messages, 'generation_response': generated,
                                      'deterministic': deterministic, 'timings': stage_timings,
                                      'metric_timings': metric_timings}
                        atomic_json(directory / 'questions' / f"q{q['question_id']}_{variant}.json", checkpoint)
                        rows.append(row)
                        manifest['completed_results'] = len(rows)
                        atomic_json(directory / 'manifest.json', manifest)
                        progress(f'{len(rows)}/{len(records)*4} complete; checkpoint saved')
            frame = pd.DataFrame(rows)
            overall, categories = summarize(frame)
            # JSON is the atomic source of truth; CSVs can be regenerated from checkpoints.
            atomic_json(directory / 'results.json', {'questions': rows,
                'overall': overall.astype(object).where(overall.notna(), None).to_dict('records'),
                'category': categories.astype(object).where(categories.notna(), None).to_dict('records')})
            for name, table in (('question_results',frame),('overall_results',overall),('category_results',categories)):
                table.to_csv(directory / (name + '.csv'), index=False)
            manifest.update(status='complete', completed_at=utc_now())
            atomic_json(directory / 'manifest.json', manifest)
            atomic_json(directory / 'attempts' / (attempt_id + '.json'), manifest)
            progress(f'EXPERIMENT3_FULL_EVALUATION_SAVE_COMPLETE: {directory}')
            return {'overall': overall, 'category': categories, 'questions': frame,
                    'manifest': manifest, 'directory': directory}
        except BaseException as error:
            failure = {**context, 'attempt_id': attempt_id, 'at': utc_now(),
                       'error': f'{type(error).__name__}: {error}'}
            if isinstance(error, GenerationFailure):
                failure['response'] = error.response
                failure['generation_attempts'] = error.attempts
            atomic_json(directory / 'failures' / (attempt_id + '.json'), failure)
            manifest.update(status='incomplete', completed_results=len(rows), last_failure=failure)
            atomic_json(directory / 'manifest.json', manifest)
            atomic_json(directory / 'attempts' / (attempt_id + '.json'), manifest)
            raise
        finally:
            await self.backend.aclose()


def summarize(frame):
    """Question-weighted answer metrics; claim-weighted retrieval; explicit N/A."""
    def mean(group, column):
        values = group[column].dropna()
        return float(values.mean()) if len(values) else None

    def row(group):
        answerable = group[group.answerable]
        unanswerable = group[~group.answerable]
        count = int(answerable.required_claim_count.sum())
        out = {'variant': group.variant.iloc[0], 'candidate_cap': int(group.candidate_cap.iloc[0]),
               'rerank': bool(group.rerank.iloc[0]), 'completed_questions': len(group), 'failed_questions': 0,
               'answerable_questions': len(answerable), 'unanswerable_questions': len(unanswerable),
               'claim_recall': float(answerable.supported_claim_count.sum()/count) if count else None,
               'legacy_claim_recall': float(answerable.legacy_supported_claim_count.sum()/count) if count else None,
               'document_hit_rate': mean(answerable,'document_hit_at_k'),
               'document_recall': mean(answerable,'document_recall_at_k'), 'mrr': mean(answerable,'reciprocal_rank'),
               'citation_validity_rate': mean(group,'citations_valid'),
               'answerable_threshold_accuracy': mean(answerable,'answer_correct'),
               'incorrect_abstention_rate': mean(answerable,'incorrect_abstention'),
               'abstention_accuracy': mean(unanswerable,'abstention_correct'),
               'overall_mixed_accuracy': mean(group,'answer_correct'),
               'mean_candidate_count': mean(group,'candidate_count'), 'mean_context_count': mean(group,'context_count'),
               'underfilled_questions': int(group.underfilled.sum()), 'empty_pools': int(group.empty_pool.sum()),
               'p95_total_latency_ms': float(group.total_latency_ms.quantile(.95)),
               'reconstructed_latency_questions': int(group.latency_provenance.eq('reconstructed_from_recorded_stages').sum())}
        for name in RAGAS_METRICS:
            out['mean_ragas_' + name] = mean(answerable,'ragas_' + name)
        for col in ('retrieval_latency_ms','reranking_latency_ms','prompt_latency_ms','generation_latency_ms',
                    'total_latency_ms','evaluation_latency_ms','evaluation_saved_ms','cache_load_ms',
                    'wall_ms_this_execution','generation_input_tokens','generation_output_tokens','generation_total_tokens',
                    'embedding_tokens','rag_total_tokens'):
            out['mean_' + col] = mean(group,col)
        return out

    overall = pd.DataFrame([row(g) for _,g in frame.groupby('variant', sort=False)])
    categories = pd.DataFrame([{**row(g), 'category': cat} for (_,cat),g in
                              frame.groupby(['variant','category'], sort=False)])
    eligible = categories[categories.category.isin(ANSWERABLE_CATEGORIES)]
    for col in ('claim_recall', *('mean_ragas_' + n for n in RAGAS_METRICS)):
        stats = eligible.groupby('variant')[col].agg(['mean','min'])
        overall['category_mean_' + col] = overall.variant.map(stats['mean'])
        overall['category_min_' + col] = overall.variant.map(stats['min'])
    return overall, categories
