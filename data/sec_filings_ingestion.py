"""Download SEC 10-K/10-Q HTML filings and store cleaned sections in Parquet.

The Parquet ``sections`` column uses Arrow's native
``list<struct<id: string, title: string, text: string>>`` type. This is a good
fit for downstream chunking and also permits 10-Q item IDs to repeat across
Part I and Part II.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from bs4.element import NavigableString
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)


TEN_K_ITEMS: Sequence[str] = (
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A",
    "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14",
    "15", "16",
)

TEN_Q_ITEMS: Mapping[str, Sequence[str]] = {
    "Part I": ("1", "2", "3", "4"),
    "Part II": ("1", "1A", "2", "3", "4", "5", "6"),
}

_ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*item\s+(?P<item>\d{1,2}[A-C]?)\s*[.\-:\u2013\u2014]?"
    r"\s*(?P<title>[^\n]{2,240})$"
)
_PART_HEADING_RE = re.compile(
    r"(?im)^\s*part\s+(?P<part>I{1,4})\b[^\n]*$"
)
_SIGNATURE_RE = re.compile(r"(?im)^\s*signatures?\s*$")
_MARKER_RE = re.compile(r"\[\[SEC_ANCHOR_(\d+)\]\]")
_SEC_ARCHIVE_RE = re.compile(
    r"^(?P<prefix>.*/Archives/edgar/data/\d+)/(?P<accession>\d{18})/[^/]+$",
    re.IGNORECASE,
)

_TITLE_HINTS: Mapping[str, Mapping[str, Sequence[str]]] = {
    "10-K": {
        "1": ("business",),
        "1A": ("risk factor",),
        "1B": ("staff comment",),
        "1C": ("cybersecurity",),
        "2": ("propert",),
        "3": ("legal proceeding",),
        "4": ("mine safety",),
        "5": ("market for", "registrant's common equity", "registrant’s common equity"),
        "6": ("reserved", "selected financial"),
        "7": ("management", "discussion and analysis"),
        "7A": ("market risk",),
        "8": ("financial statement",),
        "9": ("accountant", "accounting and financial disclosure"),
        "9A": ("controls and procedures",),
        "9B": ("other information",),
        "9C": ("foreign jurisdiction",),
        "10": ("director", "executive officer", "corporate governance"),
        "11": ("executive compensation",),
        "12": ("security ownership",),
        "13": ("relationships", "related transactions"),
        "14": ("accounting fees",),
        "15": ("exhibit", "financial statement schedules"),
        "16": ("10-k summary", "10‑k summary", "10k summary"),
    },
    "10-Q": {
        "1": ("financial statement", "legal proceeding"),
        "1A": ("risk factor",),
        "2": ("management", "discussion and analysis", "unregistered sales"),
        "3": ("market risk", "senior securities"),
        "4": ("controls and procedures", "mine safety"),
        "5": ("other information",),
        "6": ("exhibit",),
    },
}


class FilingParseError(ValueError):
    """Raised when required filing metadata or section headings are missing."""


@dataclass(frozen=True)
class _Candidate:
    start: int
    item: str
    title: str
    part: Optional[str]
    anchored: bool


class SecClient:
    """Small, rate-limited SEC HTTP client with retry handling."""

    def __init__(
        self,
        user_agent: str,
        *,
        min_request_interval: float = 0.12,
        timeout: float = 30.0,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "user_agent must identify you and include a contact email, for "
                "example 'Jane Doe jane@example.com'."
            )
        self.timeout = timeout
        self.min_request_interval = max(0.0, min_request_interval)
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        retries = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET",)),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_text(self, url: str) -> str:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        response = self.session.get(url, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        return response.text

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "SecClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _clean_inline(value: str) -> str:
    value = value.replace("\xa0", " ").replace("\u200b", "")
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([®™©])", r"\1", value)


def _xbrl_values(soup: BeautifulSoup, fact_name: str) -> List[str]:
    wanted = fact_name.casefold()
    values: List[str] = []
    for tag in soup.find_all(attrs={"name": True}):
        name = str(tag.get("name", "")).casefold()
        if name != wanted:
            continue
        value = _clean_inline(tag.get_text(" ", strip=True))
        if value and value not in values:
            values.append(value)
    return values


def _first_xbrl_value(soup: BeautifulSoup, fact_name: str) -> Optional[str]:
    values = _xbrl_values(soup, fact_name)
    return values[0] if values else None


def _iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = _clean_inline(value).rstrip(".")
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"\b(20\d{2})[-/]([01]?\d)[-/]([0-3]?\d)\b", value)
    if match:
        year, month, day = (int(group) for group in match.groups())
        return datetime(year, month, day).date().isoformat()
    return None


def _extract_metadata(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    form_type = _first_xbrl_value(soup, "dei:DocumentType")
    if form_type:
        form_type = form_type.upper().replace("FORM ", "").strip()

    ticker_values = [
        value for value in _xbrl_values(soup, "dei:TradingSymbol")
        if value.casefold() not in {"none", "n/a", "not applicable"}
    ]
    ticker = ticker_values[0] if ticker_values else None

    return {
        "company_name": _first_xbrl_value(soup, "dei:EntityRegistrantName"),
        "ticker": ticker,
        "form_type": form_type,
        "period_end": _iso_date(
            _first_xbrl_value(soup, "dei:DocumentPeriodEndDate")
        ),
    }


def _normalize_form_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = _clean_inline(value).upper().replace("FORM", "").strip()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    if normalized in {"10K", "10-K"}:
        return "10-K"
    if normalized in {"10Q", "10-Q"}:
        return "10-Q"
    return normalized


def _target_ids(soup: BeautifulSoup) -> set:
    targets = set()
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        if href.startswith("#") and len(href) > 1:
            targets.add(unquote(href[1:]))
    for tag in soup.find_all(attrs={"id": True}):
        tag_id = str(tag.get("id", ""))
        if re.search(r"(?:^|[_-])(item|part)(?:[_-]|\d|$)", tag_id, re.I):
            targets.add(tag_id)
    for tag in soup.find_all(attrs={"name": True}):
        tag_name = str(tag.get("name", ""))
        if re.search(r"(?:^|[_-])(item|part)(?:[_-]|\d|$)", tag_name, re.I):
            targets.add(tag_name)
    return targets


def _remove_non_content(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        # Descendants of a decomposed hidden parent remain in this snapshot,
        # but BeautifulSoup clears their attributes.
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        tag_name = (tag.name or "").casefold()
        style = str(tag.get("style", "")).replace(" ", "").casefold()
        if tag_name in {"ix:header", "ix:hidden"} or (
            "display:none" in style or "visibility:hidden" in style
        ):
            tag.decompose()


def _render_document(soup: BeautifulSoup) -> Tuple[str, Dict[int, str]]:
    targets = _target_ids(soup)
    marker_ids: Dict[int, str] = {}
    seen_tags = set()
    marker_number = 0

    for tag in soup.find_all(True):
        values: Iterable[str] = (
            str(tag.get("id", "")),
            str(tag.get("name", "")),
        )
        if not any(value and value in targets for value in values):
            continue
        # One marker per tag even if both id and name refer to it.
        identity = id(tag)
        if identity in seen_tags:
            continue
        seen_tags.add(identity)
        anchor_name = next(value for value in values if value and value in targets)
        marker = f"\n[[SEC_ANCHOR_{marker_number}]]\n"
        tag.insert_before(NavigableString(marker))
        marker_ids[marker_number] = anchor_name
        marker_number += 1

    _remove_non_content(soup)

    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for cell in soup.find_all(["td", "th"]):
        cell.append(NavigableString(" | "))
    for row in soup.find_all("tr"):
        row.append(NavigableString("\n"))
    for block in soup.find_all(
        ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        block.append(NavigableString("\n"))

    root = soup.body or soup
    text = root.get_text(" ", strip=False)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, marker_ids


def _part_at(text: str, position: int) -> Optional[str]:
    latest: Optional[str] = None
    for match in _PART_HEADING_RE.finditer(text, 0, position):
        roman = match.group("part").upper()
        if roman in {"I", "II"}:
            latest = f"Part {roman}"
    return latest


def _plausible_heading(form_type: str, item: str, title: str) -> bool:
    form_type = form_type.upper()
    item = item.upper()
    valid_items = set(TEN_K_ITEMS) if form_type == "10-K" else {
        value for values in TEN_Q_ITEMS.values() for value in values
    }
    if item not in valid_items:
        return False
    normalized_title = _clean_inline(title).casefold().strip(".|:- ")
    hints = _TITLE_HINTS.get(form_type, {}).get(item, ())
    return bool(normalized_title) and (
        not hints or any(hint.casefold() in normalized_title for hint in hints)
    )


def _candidate_from_tail(
    text: str,
    marker_end: int,
    form_type: str,
) -> Optional[Tuple[str, str]]:
    tail = _MARKER_RE.sub("", text[marker_end:marker_end + 700])
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    # Some filings put "Item 1." and its title on adjacent lines.
    choices = [lines[0]]
    if len(lines) > 1:
        choices.append(f"{lines[0]} {lines[1]}")
    for choice in choices:
        match = _ITEM_HEADING_RE.match(choice)
        if not match:
            continue
        item = match.group("item").upper()
        title = match.group("title").strip()
        if _plausible_heading(form_type, item, title):
            return item, title
    return None


def _find_candidates(text: str, form_type: str) -> List[_Candidate]:
    candidates: List[_Candidate] = []

    for marker in _MARKER_RE.finditer(text):
        parsed = _candidate_from_tail(text, marker.end(), form_type)
        if not parsed:
            continue
        item, title = parsed
        candidates.append(
            _Candidate(
                start=marker.start(),
                item=item,
                title=title,
                part=_part_at(text, marker.start()) if form_type == "10-Q" else None,
                anchored=True,
            )
        )

    # Text headings are a fallback for filings without usable HTML anchors and
    # also fill isolated gaps when only some headings have anchors.
    for match in _ITEM_HEADING_RE.finditer(text):
        item = match.group("item").upper()
        title = match.group("title").strip()
        if not _plausible_heading(form_type, item, title):
            continue
        candidates.append(
            _Candidate(
                start=match.start(),
                item=item,
                title=title,
                part=_part_at(text, match.start()) if form_type == "10-Q" else None,
                anchored=False,
            )
        )

    candidates.sort(key=lambda candidate: (candidate.start, not candidate.anchored))
    deduped: List[_Candidate] = []
    for candidate in candidates:
        if deduped:
            prior = deduped[-1]
            same_key = (prior.item, prior.part) == (candidate.item, candidate.part)
            if same_key and candidate.start - prior.start < 500:
                # Prefer the earlier anchored marker; it normally sits directly
                # before the visible heading and includes the complete title.
                if candidate.anchored and not prior.anchored:
                    deduped[-1] = candidate
                continue
        deduped.append(candidate)
    return deduped


def _section_key(candidate: _Candidate, form_type: str) -> Optional[str]:
    if form_type == "10-K":
        return f"Item {candidate.item}"
    if candidate.part not in TEN_Q_ITEMS:
        return None
    if candidate.item not in TEN_Q_ITEMS[candidate.part]:
        return None
    return f"{candidate.part} Item {candidate.item}"


def _clean_section(text: str) -> str:
    text = _MARKER_RE.sub("", text)
    lines = []
    previous = None
    for raw_line in text.splitlines():
        line = _clean_inline(raw_line).strip("|").strip()
        if not line or re.fullmatch(r"\d{1,3}", line) or line == "* * *":
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines).strip()


def _extract_sections(text: str, form_type: str) -> List[Dict[str, str]]:
    candidates = _find_candidates(text, form_type)
    if not candidates:
        raise FilingParseError("No SEC item headings were found in the filing HTML.")

    structural_ends = [match.start() for match in _PART_HEADING_RE.finditer(text)]
    structural_ends.extend(match.start() for match in _SIGNATURE_RE.finditer(text))
    structural_ends.sort()

    choices: Dict[str, Tuple[int, bool, str, str, str]] = {}
    for index, candidate in enumerate(candidates):
        key = _section_key(candidate, form_type)
        if key is None:
            continue
        end = candidates[index + 1].start if index + 1 < len(candidates) else len(text)
        for boundary in structural_ends:
            if candidate.start < boundary < end:
                end = boundary
                break
        section = _clean_section(text[candidate.start:end])
        section_lines = section.splitlines()
        if section_lines and _ITEM_HEADING_RE.match(section_lines[0]):
            section = "\n".join(section_lines[1:]).strip()
        # Empty anchored items (for example Item 6, "Reserved") are still
        # meaningful. Empty unanchored matches are more likely TOC noise.
        if not section and not candidate.anchored:
            continue

        # TOC entries and cross-references are usually much shorter than the
        # real item. Prefer anchored candidates, then the longest candidate.
        score = len(section) + (10_000_000 if candidate.anchored else 0)
        existing = choices.get(key)
        if existing is None or score > existing[0]:
            title = _clean_inline(candidate.title).strip(" .|:-")
            choices[key] = (
                score,
                candidate.anchored,
                candidate.item,
                title,
                section,
            )

    order: List[str]
    if form_type == "10-K":
        order = [f"Item {item}" for item in TEN_K_ITEMS]
    else:
        order = [
            f"{part} Item {item}"
            for part, items in TEN_Q_ITEMS.items()
            for item in items
        ]
    sections = [
        {
            "id": choices[key][2],
            "title": choices[key][3],
            "text": choices[key][4],
        }
        for key in order
        if key in choices
    ]
    if not sections:
        raise FilingParseError("Item headings were found, but no sections could be extracted.")
    return sections


def parse_filing_html(
    html: str,
    *,
    filing_date: Optional[str] = None,
    period_end: Optional[str] = None,
    ticker: Optional[str] = None,
    form_type: Optional[str] = None,
) -> Dict[str, object]:
    """Parse one SEC filing HTML document into the requested row schema.

    ``filing_date`` is normally supplied from the filing's SEC detail page by
    :func:`filings_to_parquet`; it is an acceptance attribute, not a dependable
    inline-XBRL document fact.
    """

    # Inline-XBRL documents are XHTML-like but are served and rendered as HTML.
    # BeautifulSoup warns about that hybrid even though the HTML parser is the
    # correct tolerant choice for real EDGAR filings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    metadata = _extract_metadata(soup)
    parsed_form_type = _normalize_form_type(
        metadata.get("form_type") or form_type
    )
    metadata["ticker"] = metadata.get("ticker") or _clean_inline(ticker or "") or None
    metadata["period_end"] = metadata.get("period_end") or _iso_date(period_end)
    if parsed_form_type not in {"10-K", "10-Q"}:
        raise FilingParseError(
            f"Expected a 10-K or 10-Q, found {parsed_form_type!r}."
        )
    missing = [
        key for key in ("company_name", "ticker", "period_end")
        if not metadata.get(key)
    ]
    if missing:
        raise FilingParseError(f"Missing required filing metadata: {missing}")

    text, _ = _render_document(soup)
    sections = _extract_sections(text, parsed_form_type)
    return {
        "company_name": metadata["company_name"],
        "ticker": metadata["ticker"],
        "form_type": parsed_form_type,
        "filing_date": _iso_date(filing_date),
        "period_end": metadata["period_end"],
        "sections": sections,
    }


def filing_index_url(document_url: str) -> Optional[str]:
    """Return the SEC filing-detail URL associated with a document URL."""

    parsed = urlparse(document_url)
    match = _SEC_ARCHIVE_RE.match(parsed.path)
    if not match:
        return None
    accession = match.group("accession")
    formatted = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    path = f"{match.group('prefix')}/{accession}/{formatted}-index.html"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def parse_filing_date(index_html: str) -> Optional[str]:
    """Extract the filing date from an SEC filing-detail HTML page."""

    return parse_filing_details(index_html)["filing_date"]


def parse_filing_details(index_html: str) -> Dict[str, Optional[str]]:
    """Extract authoritative dates from an SEC filing-detail HTML page."""

    soup = BeautifulSoup(index_html, "lxml")
    details: Dict[str, Optional[str]] = {
        "filing_date": None,
        "period_end": None,
    }
    labels = {
        "filing date": "filing_date",
        "period of report": "period_end",
    }
    for head in soup.find_all(class_="infoHead"):
        label = _clean_inline(head.get_text(" ", strip=True)).casefold()
        key = labels.get(label)
        if key is None:
            continue
        info = head.find_next_sibling(class_="info")
        if info:
            details[key] = _iso_date(info.get_text(" ", strip=True))
    if not details["filing_date"]:
        match = re.search(
            r"Filing\s+Date\s+(20\d{2}-\d{2}-\d{2})",
            soup.get_text(" "),
        )
        details["filing_date"] = match.group(1) if match else None
    return details


def _arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("company_name", pa.string(), nullable=False),
            pa.field("ticker", pa.string(), nullable=False),
            pa.field("form_type", pa.string(), nullable=False),
            pa.field("filing_date", pa.string(), nullable=True),
            pa.field("period_end", pa.string(), nullable=False),
            pa.field(
                "sections",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("id", pa.string(), nullable=False),
                            pa.field("title", pa.string(), nullable=False),
                            pa.field("text", pa.string(), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
        ]
    )


def filings_to_parquet(
    filings: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    url_column: str = "url",
    user_agent: str,
    filing_date_column: Optional[str] = None,
    ticker_column: Optional[str] = None,
    form_type_column: Optional[str] = None,
    compression: str = "zstd",
    min_request_interval: float = 0.12,
) -> Path:
    """Download filing URLs from a DataFrame and write one row per filing.

    Parameters
    ----------
    filings:
        A DataFrame containing one SEC filing-document URL per row.
    output_path:
        Destination ``.parquet`` file.
    url_column:
        Name of the DataFrame column containing the filing URLs.
    user_agent:
        SEC-compliant identity including a contact email.
    filing_date_column:
        Optional input column containing ISO filing dates. If absent or null,
        the authoritative date is fetched from each SEC filing-detail page.
    ticker_column:
        Optional input column used when a filing omits its inline-XBRL ticker.
    form_type_column:
        Optional input column used when a filing omits its inline-XBRL form.
    compression:
        Arrow Parquet compression codec; ``zstd`` is a good default for text.
    min_request_interval:
        Delay between SEC requests. The default stays below 10 requests/second.

    Returns
    -------
    pathlib.Path
        The written Parquet path.
    """

    if not isinstance(filings, pd.DataFrame):
        raise TypeError("filings must be a pandas DataFrame")
    if url_column not in filings.columns:
        raise KeyError(f"DataFrame has no URL column named {url_column!r}")
    if filing_date_column and filing_date_column not in filings.columns:
        raise KeyError(
            f"DataFrame has no filing-date column named {filing_date_column!r}"
        )
    for column_name, purpose in (
        (ticker_column, "ticker"),
        (form_type_column, "form type"),
    ):
        if column_name and column_name not in filings.columns:
            raise KeyError(
                f"DataFrame has no {purpose} column named {column_name!r}"
            )
    if filings.empty:
        raise ValueError("filings DataFrame is empty")

    rows: List[Dict[str, object]] = []
    total = len(filings)
    with SecClient(
        user_agent,
        min_request_interval=min_request_interval,
    ) as client:
        for row_number, (_, input_row) in enumerate(filings.iterrows(), start=1):
            url = str(input_row[url_column]).strip()
            if not url or url.casefold() == "nan":
                raise ValueError(f"Row {row_number} has an empty filing URL")
            LOGGER.info("Processing filing %d/%d: %s", row_number, total, url)

            supplied_date: Optional[str] = None
            if filing_date_column:
                value = input_row[filing_date_column]
                if pd.notna(value):
                    supplied_date = _iso_date(str(value))

            filing_date = supplied_date
            period_end: Optional[str] = None
            if not filing_date:
                index_url = filing_index_url(url)
                if not index_url:
                    raise FilingParseError(
                        f"Cannot derive an SEC filing-detail URL from {url!r}; "
                        "provide filing_date_column instead."
                    )
                details = parse_filing_details(client.get_text(index_url))
                filing_date = details["filing_date"]
                period_end = details["period_end"]
                if not filing_date:
                    raise FilingParseError(
                        f"Could not extract Filing Date from {index_url!r}"
                    )

            html = client.get_text(url)
            ticker = None
            if ticker_column and pd.notna(input_row[ticker_column]):
                ticker = str(input_row[ticker_column])
            input_form_type = None
            if form_type_column and pd.notna(input_row[form_type_column]):
                input_form_type = str(input_row[form_type_column])
            rows.append(
                parse_filing_html(
                    html,
                    filing_date=filing_date,
                    period_end=period_end,
                    ticker=ticker,
                    form_type=input_form_type,
                )
            )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_arrow_schema())
    pq.write_table(table, output, compression=compression)
    return output


def read_filings_parquet(path: Union[str, Path]) -> List[Dict[str, object]]:
    """Read the Parquet file as normal Python dictionaries and lists."""

    return pq.read_table(path).to_pylist()


DEFAULT_INPUT_PATH = Path(__file__).resolve().with_name("url_links.csv")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name(
    "sec_corpus_filings.parquet"
)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SEC 10-K/10-Q filing URLs to a Parquet corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output Parquet file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--url-column",
        default="Link",
        help="CSV column containing SEC document URLs (default: Link)",
    )
    parser.add_argument(
        "--filing-date-column",
        default=None,
        help="Optional CSV column containing authoritative filing dates",
    )
    parser.add_argument(
        "--ticker-column",
        default="Ticker",
        help="Optional CSV ticker fallback column (default: Ticker)",
    )
    parser.add_argument(
        "--form-type-column",
        default="Type",
        help="Optional CSV form-type fallback column (default: Type)",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help=(
            "SEC identity with contact email. Alternatively set "
            "SEC_USER_AGENT."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (useful for a smoke test)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point. Importing this module performs no downloads."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if not args.user_agent:
        parser.error(
            "provide --user-agent 'Your Name you@example.com' or set "
            "SEC_USER_AGENT"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.input.is_file():
        parser.error(f"input CSV does not exist: {args.input}")

    filings = pd.read_csv(args.input)
    if args.limit is not None:
        filings = filings.head(args.limit)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ticker_column = args.ticker_column if args.ticker_column in filings.columns else None
    form_type_column = (
        args.form_type_column if args.form_type_column in filings.columns else None
    )
    output = filings_to_parquet(
        filings,
        args.output,
        url_column=args.url_column,
        user_agent=args.user_agent,
        filing_date_column=args.filing_date_column,
        ticker_column=ticker_column,
        form_type_column=form_type_column,
    )
    LOGGER.info("Wrote %d filings to %s", len(filings), output)
    return 0


__all__ = [
    "FilingParseError",
    "SecClient",
    "filing_index_url",
    "filings_to_parquet",
    "parse_filing_date",
    "parse_filing_details",
    "parse_filing_html",
    "read_filings_parquet",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
