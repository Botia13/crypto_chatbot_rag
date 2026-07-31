# crypto_chatbot_rag

Chatbot made with a RAG implementation for researching crypto-exposed public
companies.

## SEC filing ingestion

Create and activate a virtual environment, then install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The SEC requires automated clients to identify themselves. Run the ingestion
script from any working directory with your real name and contact email:

```bash
python data/sec_filings_ingestion.py \
  --user-agent "Your Name your.email@example.com"
```

By default, the script reads `data/url_links.csv` and writes
`data/sec_corpus_filings.parquet`. Test the first two filings before processing
the complete corpus with:

```bash
python data/sec_filings_ingestion.py \
  --user-agent "Your Name your.email@example.com" \
  --limit 2 \
  --output /tmp/sec_filings_smoke_test.parquet
```

You may use the `SEC_USER_AGENT` environment variable instead of passing the
command-line option.

The Parquet `sections` field has the native Arrow type
`list<struct<id: string, title: string, text: string>>`.

Run the regression tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
