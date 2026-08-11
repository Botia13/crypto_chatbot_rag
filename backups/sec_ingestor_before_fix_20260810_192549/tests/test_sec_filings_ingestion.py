from pathlib import Path
import sys

import pyarrow as pa
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "input_data_ingestion"))

import sec_filings_ingestion as ingestion  # noqa: E402


def _render_and_extract(html: str, form_type: str):
    soup = BeautifulSoup(html, "lxml")
    text, _, tables = ingestion._render_document(soup)
    return ingestion._extract_sections(text, form_type, tables)


def test_financial_table_preserves_rows_columns_spans_and_repeated_values():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 8. Financial Statements and Supplementary Data</h1>
          <table>
            <tr><th rowspan="2">Metric</th><th colspan="2">December 31</th></tr>
            <tr><th>2024</th><th>2023</th></tr>
            <tr><td><div>Net asset value</div></td><td>$ 53.09</td><td>$ 25.00</td></tr>
            <tr><td>Zero values</td><td>0</td><td>0</td></tr>
            <tr><td>Integer values</td><td>25</td><td>100</td></tr>
          </table>
          <h1>Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure</h1>
          <p>None.</p>
        </body></html>
        """,
        "10-K",
    )

    financials = sections[0]
    assert financials["id"] == "8"
    assert financials["part"] is None
    assert financials["text"] == "[[TABLE:table-0001]]"
    assert "Net asset value | $ 53.09 | $ 25.00" not in financials["text"]
    assert financials["tables"][0]["rows"] == [
        ["Metric", "December 31", ""],
        ["Metric", "2024", "2023"],
        ["Net asset value", "$ 53.09", "$ 25.00"],
        ["Zero values", "0", "0"],
        ["Integer values", "25", "100"],
    ]


def test_ten_q_repeated_item_ids_keep_their_part_identity():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Part I</h1>
          <h2>Item 1. Financial Statements</h2><p>Financial content.</p>
          <h2>Item 2. Management's Discussion and Analysis</h2><p>Analysis.</p>
          <h1>Part II</h1>
          <h2>Item 1. Legal Proceedings</h2><p>None.</p>
          <h2>Item 1A. Risk Factors</h2><p>Risk content.</p>
        </body></html>
        """,
        "10-Q",
    )

    item_ones = [section for section in sections if section["id"] == "1"]
    assert [(section["part"], section["title"]) for section in item_ones] == [
        ("Part I", "Financial Statements"),
        ("Part II", "Legal Proceedings"),
    ]


def test_source_url_produces_accession_and_cik():
    result = ingestion.parse_filing_html(
        """
        <html><body>
          <ix:nonNumeric name="dei:EntityRegistrantName">Example Trust</ix:nonNumeric>
          <ix:nonNumeric name="dei:TradingSymbol">TEST</ix:nonNumeric>
          <ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>
          <ix:nonNumeric name="dei:DocumentPeriodEndDate">2024-12-31</ix:nonNumeric>
          <h1>Item 1. Business</h1><p>Example business.</p>
        </body></html>
        """,
        filing_date="2025-02-01",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1234567/"
            "000123456725000001/example.htm"
        ),
    )

    assert result["source_url"].endswith("example.htm")
    assert result["accession_number"] == "0001234567-25-000001"
    assert result["cik"] == "1234567"


def test_arrow_schema_contains_structured_tables_and_provenance():
    schema = ingestion._arrow_schema()
    assert {"source_url", "accession_number", "cik"} <= set(schema.names)
    section_type = schema.field("sections").type.value_type
    assert section_type.field("part").nullable
    table_type = section_type.field("tables").type.value_type
    assert pa.types.is_list(table_type.field("rows").type)
