from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from RAG_model.ingestion import sec_filings_ingestion as ingestion  # noqa: E402


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
    assert financials["section_type"] == "item"
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
    assert section_type.field("section_type").type == pa.string()
    table_type = section_type.field("tables").type.value_type
    assert pa.types.is_list(table_type.field("rows").type)
    assert table_type.field("table_type").type == pa.string()
    assert table_type.field("table_title").nullable


def test_split_inline_heading_selects_body_instead_of_table_of_contents():
    sections = _render_and_extract(
        """
        <html><body>
          <div>Part I - Financial Information</div>
          <div>Item 1. Financial Statements (Unaudited) | 4</div>
          <div>Item 2. Management's Discussion and Analysis | 19</div>
          <h1>Part I - Financial Information</h1>
          <a href="#item1">Financial statements</a>
          <h2 id="item1">Item 1. Financial St<span>atements</span> (Unaudited)</h2>
          <p>GRAYSCALE TEST TRUST</p>
          <table>
            <tr><th>Assets</th><th>2025</th><th>2024</th></tr>
            <tr><td>Total assets</td><td>$ 100</td><td>$ 80</td></tr>
          </table>
          <h2>Item 2. Management's Discussion and Analysis</h2>
          <p>Analysis body.</p>
          <h1>Part II - Other Information</h1>
          <h2>Item 1. Legal Proceedings</h2><p>None.</p>
        </body></html>
        """,
        "10-Q",
    )

    financials = next(
        section for section in sections
        if section["part"] == "Part I" and section["id"] == "1"
    )
    assert financials["title"] == "Financial Statements"
    assert "GRAYSCALE TEST TRUST" in financials["text"]
    assert len(financials["tables"]) == 1
    assert financials["tables"][0]["table_id"] == "table-0001"


def test_financial_appendix_after_signatures_is_retained_as_own_section():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 8. Financial Statements and Supplementary Data</h1>
          <p>See Index to Financial Statements.</p>
          <h1>Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure</h1>
          <p>None.</p>
          <h1>Item 15. Exhibits and Financial Statement Schedules</h1>
          <p>Exhibits.</p>
          <h1>Item 16. Form 10-K Summary</h1><p>None.</p>
          <h1>SIGNATURES</h1><p>Signed.</p>
          <h1>INDEX TO FINANCIAL STATEMENTS</h1>
          <p>Statements of Assets and Liabilities</p>
          <table>
            <tr><th>Assets</th><th>2025</th></tr>
            <tr><td>Total assets</td><td>$ 100</td></tr>
          </table>
        </body></html>
        """,
        "10-K",
    )

    item_16 = next(section for section in sections if section["id"] == "16")
    appendix = next(
        section for section in sections
        if section["id"] == "FINANCIAL_STATEMENTS"
    )
    assert item_16["tables"] == []
    assert appendix["section_type"] == "financial_statements"
    assert len(appendix["tables"]) == 1


def test_bullet_layout_table_is_flattened_into_narrative():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 1A. Risk Factors</h1>
          <table role="presentation">
            <tr><td>●</td><td>Loss of private keys could cause permanent loss.</td></tr>
          </table>
          <h1>Item 2. Properties</h1><p>None.</p>
        </body></html>
        """,
        "10-K",
    )

    risks = next(section for section in sections if section["id"] == "1A")
    assert risks["tables"] == []
    assert "Loss of private keys" in risks["text"]


def test_table_ids_follow_document_order_and_include_context():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 8. Financial Statements and Supplementary Data</h1>
          <h2>Statements of Assets and Liabilities</h2>
          <p>(Amounts in thousands of US dollars)</p>
          <table>
            <tr><th>Assets</th><th>2025</th></tr>
            <tr><td>Total assets</td><td>$ 100</td></tr>
          </table>
          <h2>Statements of Operations</h2>
          <table>
            <tr><th>Operations</th><th>2025</th></tr>
            <tr><td>Net income</td><td>$ 20</td></tr>
          </table>
          <h1>Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure</h1>
          <p>None.</p>
        </body></html>
        """,
        "10-K",
    )

    tables = next(section for section in sections if section["id"] == "8")["tables"]
    assert [table["table_id"] for table in tables] == ["table-0001", "table-0002"]
    assert tables[0]["table_type"] == "financial_table"
    assert "Assets and Liabilities" in (tables[0]["table_title"] or "")
    assert tables[0]["embedding_text"] == "Assets | 2025\nTotal assets | $ 100"


def test_principal_accountant_fees_is_not_absorbed_by_item_13():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 13. Certain Relationships and Related Transactions, and Director Independence</h1>
          <p>No related transactions.</p>
          <h1>Item 14. Principal Accountant Fees and Services</h1>
          <table>
            <tr><th>Fee</th><th>2025</th></tr>
            <tr><td>Audit fees</td><td>$ 100</td></tr>
          </table>
          <h1>Item 15. Exhibits and Financial Statement Schedules</h1>
          <p>Exhibits.</p>
        </body></html>
        """,
        "10-K",
    )

    item_13 = next(section for section in sections if section["id"] == "13")
    item_14 = next(section for section in sections if section["id"] == "14")
    assert item_13["tables"] == []
    assert item_14["title"] == "Principal Accountant Fees and Services"
    assert len(item_14["tables"]) == 1


def test_glossary_is_separate_from_item_16():
    sections = _render_and_extract(
        """
        <html><body>
          <h1>Item 15. Exhibits and Financial Statement Schedules</h1>
          <p>Exhibits.</p>
          <h1>Item 16. Form 10-K Summary</h1><p>None.</p>
          <h1>GLOSSARY OF DEFINED TERMS</h1>
          <p>Custodian means the entity safeguarding the Trust's assets.</p>
          <h1>SIGNATURES</h1><p>Signed.</p>
        </body></html>
        """,
        "10-K",
    )

    item_16 = next(section for section in sections if section["id"] == "16")
    glossary = next(section for section in sections if section["id"] == "GLOSSARY")
    assert "Custodian means" not in item_16["text"]
    assert glossary["section_type"] == "glossary"
    assert "Custodian means" in glossary["text"]


def test_new_schema_parquet_and_chunking_roundtrip(tmp_path):
    document = ingestion.parse_filing_html(
        """
        <html><body>
          <ix:nonNumeric name="dei:EntityRegistrantName">Example Trust</ix:nonNumeric>
          <ix:nonNumeric name="dei:TradingSymbol">TEST</ix:nonNumeric>
          <ix:nonNumeric name="dei:DocumentType">10-Q</ix:nonNumeric>
          <ix:nonNumeric name="dei:DocumentPeriodEndDate">2025-03-31</ix:nonNumeric>
          <h1>Part I</h1>
          <h2>Item 1. Financial Statements</h2>
          <h3>Statements of Assets and Liabilities</h3>
          <table>
            <tr><th>Assets</th><th>2025</th></tr>
            <tr><td>Total assets</td><td>$ 100</td></tr>
          </table>
          <h2>Item 2. Management's Discussion and Analysis</h2><p>Analysis.</p>
          <h1>Part II</h1>
          <h2>Item 1. Legal Proceedings</h2><p>None.</p>
        </body></html>
        """,
        filing_date="2025-05-01",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1234567/"
            "000123456725000001/example.htm"
        ),
    )
    output = tmp_path / "roundtrip.parquet"
    pq.write_table(
        pa.Table.from_pylist([document], schema=ingestion._arrow_schema()),
        output,
    )
    restored = pq.read_table(output).to_pylist()[0]
    table = restored["sections"][0]["tables"][0]
    assert table["table_type"] == "financial_table"
    assert table["embedding_text"] == "Assets | 2025\nTotal assets | $ 100"

    from RAG_model.ingestion.chunking import chunk_document

    chunks = chunk_document(restored)
    table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
    assert table_chunk["section_type"] == "item"
    assert table_chunk["section_key"] == "part-i-item-1"
    assert table_chunk["table_type"] == "financial_table"
    assert "Table title: Statements of Assets and Liabilities" in table_chunk["chunk_text"]
    assert "Total assets | $ 100" in table_chunk["chunk_text"]
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert any(chunk["section_key"] == "part-ii-item-1" for chunk in chunks)
