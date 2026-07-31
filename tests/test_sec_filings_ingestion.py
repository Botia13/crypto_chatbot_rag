from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import sec_filings_ingestion as ingestion


def test_import_has_no_download_side_effect(tmp_path: Path) -> None:
    # Reaching this test proves importing the module did not read the CSV or
    # start the 75-filing network job that used to live at module scope.
    assert callable(ingestion.filings_to_parquet)
    assert not (tmp_path / "sec_corpus_filings.parquet").exists()


def test_default_paths_are_relative_to_script_not_cwd() -> None:
    data_dir = Path(ingestion.__file__).resolve().parent
    assert ingestion.DEFAULT_INPUT_PATH == data_dir / "url_links.csv"
    assert ingestion.DEFAULT_OUTPUT_PATH == data_dir / "sec_corpus_filings.parquet"


def test_nested_sections_arrow_schema_round_trip(tmp_path: Path) -> None:
    row = {
        "company_name": "Example Trust",
        "ticker": "EX",
        "form_type": "10-K",
        "filing_date": "2026-01-02",
        "period_end": "2025-12-31",
        "sections": [
            {"id": "1", "title": "Business", "text": "Example text"},
            {"id": "1A", "title": "Risk Factors", "text": "Risk text"},
        ],
    }
    output = tmp_path / "round_trip.parquet"
    table = pa.Table.from_pylist([row], schema=ingestion._arrow_schema())
    pq.write_table(table, output)

    restored = ingestion.read_filings_parquet(output)
    assert restored == [row]
    assert pa.types.is_list(pq.read_schema(output).field("sections").type)


def test_filing_details_supply_period_end_fallback() -> None:
    index_html = """
    <div class="infoHead">Filing Date</div><div class="info">2026-02-25</div>
    <div class="infoHead">Period of Report</div><div class="info">2025-12-31</div>
    """
    assert ingestion.parse_filing_details(index_html) == {
        "filing_date": "2026-02-25",
        "period_end": "2025-12-31",
    }


def test_cli_rejects_missing_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    parser = ingestion._build_argument_parser()
    args = parser.parse_args([])
    assert args.user_agent is None


def test_dataframe_requires_configured_url_column(tmp_path: Path) -> None:
    frame = pd.DataFrame({"Link": ["https://www.sec.gov/example.htm"]})
    with pytest.raises(KeyError, match="URL column"):
        ingestion.filings_to_parquet(
            frame,
            tmp_path / "out.parquet",
            url_column="url",
            user_agent="Example User user@example.com",
        )
