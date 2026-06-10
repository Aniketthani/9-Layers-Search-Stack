"""
Module: document_ingestion.py
Handles extraction of raw text + structured chunks from:
PDF, DOCX, XLSX, .msg/.eml email, HTML (ACORD-style)
Each chunk carries position metadata for downstream use.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


@dataclass
class DocumentChunk:
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    source_type: str = ""       # pdf, docx, xlsx, email, acord
    table_data: Optional[list] = None   # for tabular chunks
    raw_metadata: dict = field(default_factory=dict)


@dataclass
class IngestedDocument:
    filename: str
    source_type: str
    chunks: List[DocumentChunk]
    full_text: str
    metadata: dict = field(default_factory=dict)


def ingest_document(file_path: str) -> IngestedDocument:
    path = Path(file_path)
    ext = path.suffix.lower()

    logger.info(f"Ingesting {path.name} as {ext}")

    if ext == ".pdf":
        return _ingest_pdf(path)
    elif ext == ".docx":
        return _ingest_docx(path)
    elif ext in (".xlsx", ".xls"):
        return _ingest_excel(path)
    elif ext in (".msg", ".eml"):
        return _ingest_email(path)
    elif ext in (".html", ".htm", ".xml"):
        return _ingest_html_acord(path)
    elif ext == ".txt":
        return _ingest_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── PDF ────────────────────────────────────────────────────────────────────────

def _ingest_pdf(path: Path) -> IngestedDocument:
    import pdfplumber

    chunks = []
    full_text_parts = []
    chunk_index = 0

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables first
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                table_text = _table_to_text(table)
                chunks.append(DocumentChunk(
                    text=table_text,
                    page=page_num,
                    section="table",
                    chunk_index=chunk_index,
                    source_type="pdf",
                    table_data=table,
                ))
                chunk_index += 1
                full_text_parts.append(table_text)

            # Extract paragraphs from page text
            text = page.extract_text() or ""
            paragraphs = _split_paragraphs(text)
            for para in paragraphs:
                if len(para.strip()) < 10:
                    continue
                chunks.append(DocumentChunk(
                    text=para,
                    page=page_num,
                    section=_guess_section(para),
                    chunk_index=chunk_index,
                    source_type="pdf",
                ))
                chunk_index += 1
                full_text_parts.append(para)

    return IngestedDocument(
        filename=path.name,
        source_type="pdf",
        chunks=chunks,
        full_text="\n".join(full_text_parts),
        metadata={"pages": len(chunks)},
    )


# ── DOCX ───────────────────────────────────────────────────────────────────────

def _ingest_docx(path: Path) -> IngestedDocument:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(str(path))
    chunks = []
    full_text_parts = []
    chunk_index = 0
    current_section = "body"

    for block in doc.element.body:
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

        if tag == "p":
            from docx.text.paragraph import Paragraph as DocxPara
            para = DocxPara(block, doc)
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style else ""
            if "Heading" in style:
                current_section = text
            chunks.append(DocumentChunk(
                text=text,
                section=current_section,
                chunk_index=chunk_index,
                source_type="docx",
                raw_metadata={"style": style},
            ))
            chunk_index += 1
            full_text_parts.append(text)

        elif tag == "tbl":
            from docx.table import Table as DocxTable
            tbl = DocxTable(block, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
            table_text = _table_to_text(rows)
            chunks.append(DocumentChunk(
                text=table_text,
                section=current_section,
                chunk_index=chunk_index,
                source_type="docx",
                table_data=rows,
            ))
            chunk_index += 1
            full_text_parts.append(table_text)

    return IngestedDocument(
        filename=path.name,
        source_type="docx",
        chunks=chunks,
        full_text="\n".join(full_text_parts),
    )


# ── EXCEL ──────────────────────────────────────────────────────────────────────

def _ingest_excel(path: Path) -> IngestedDocument:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    chunks = []
    full_text_parts = []
    chunk_index = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cleaned = [str(c).strip() if c is not None else "" for c in row]
            if any(cleaned):
                rows.append(cleaned)

        if not rows:
            continue

        table_text = _table_to_text(rows)
        # Also produce a flat prose version for keyword matching
        prose = " | ".join([" ".join(r) for r in rows])

        chunks.append(DocumentChunk(
            text=prose,
            section=f"sheet:{sheet_name}",
            chunk_index=chunk_index,
            source_type="xlsx",
            table_data=rows,
            raw_metadata={"sheet": sheet_name},
        ))
        chunk_index += 1
        full_text_parts.append(table_text)

    return IngestedDocument(
        filename=path.name,
        source_type="xlsx",
        chunks=chunks,
        full_text="\n".join(full_text_parts),
    )


# ── EMAIL ──────────────────────────────────────────────────────────────────────

def _ingest_email(path: Path) -> IngestedDocument:
    chunks = []
    full_text_parts = []
    chunk_index = 0

    if path.suffix.lower() == ".msg":
        try:
            import extract_msg
            msg = extract_msg.Message(str(path))
            subject = msg.subject or ""
            body = msg.body or ""
            sender = msg.sender or ""
            metadata = {"subject": subject, "sender": sender}
        except Exception as e:
            logger.warning(f"extract_msg failed: {e}, falling back to raw read")
            body = path.read_text(errors="ignore")
            metadata = {}
    else:
        import email as email_lib
        raw = path.read_bytes()
        msg = email_lib.message_from_bytes(raw)
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        metadata = {"subject": subject, "sender": sender}

    for para in _split_paragraphs(body):
        if len(para.strip()) < 10:
            continue
        chunks.append(DocumentChunk(
            text=para,
            section="email_body",
            chunk_index=chunk_index,
            source_type="email",
        ))
        chunk_index += 1
        full_text_parts.append(para)

    return IngestedDocument(
        filename=path.name,
        source_type="email",
        chunks=chunks,
        full_text="\n".join(full_text_parts),
        metadata=metadata,
    )


# ── HTML / ACORD ───────────────────────────────────────────────────────────────

def _ingest_html_acord(path: Path) -> IngestedDocument:
    from bs4 import BeautifulSoup

    raw = path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    chunks = []
    full_text_parts = []
    chunk_index = 0

    # Extract tables (ACORD forms are heavily tabular)
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(row):
                rows.append(row)
        if rows:
            table_text = _table_to_text(rows)
            chunks.append(DocumentChunk(
                text=table_text,
                section="acord_table",
                chunk_index=chunk_index,
                source_type="acord",
                table_data=rows,
            ))
            chunk_index += 1
            full_text_parts.append(table_text)

    # Extract paragraphs outside tables
    for tag in soup.find_all(["p", "div", "span", "li"]):
        text = tag.get_text(strip=True)
        if len(text) < 10:
            continue
        chunks.append(DocumentChunk(
            text=text,
            section=_guess_section(text),
            chunk_index=chunk_index,
            source_type="acord",
        ))
        chunk_index += 1
        full_text_parts.append(text)

    return IngestedDocument(
        filename=path.name,
        source_type="acord",
        chunks=chunks,
        full_text="\n".join(full_text_parts),
    )


# ── TXT ────────────────────────────────────────────────────────────────────────

def _ingest_txt(path: Path) -> IngestedDocument:
    text = path.read_text(errors="ignore")
    chunks = []
    for i, para in enumerate(_split_paragraphs(text)):
        if len(para.strip()) < 10:
            continue
        chunks.append(DocumentChunk(
            text=para,
            section=_guess_section(para),
            chunk_index=i,
            source_type="txt",
        ))
    return IngestedDocument(
        filename=path.name,
        source_type="txt",
        chunks=chunks,
        full_text=text,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n{2,}", text)
    result = []
    for p in paras:
        p = p.strip()
        if p:
            result.append(p)
    return result


def _table_to_text(table: list) -> str:
    lines = []
    for row in table:
        if row:
            lines.append(" | ".join([str(c) if c else "" for c in row]))
    return "\n".join(lines)


SECTION_KEYWORDS = {
    "risk description": ["risk description", "property description", "risk details", "insured premises"],
    "prior claims": ["prior claims", "loss history", "claims history", "previous claims"],
    "operations": ["business operations", "operations", "activities", "occupancy"],
    "general conditions": ["general conditions", "exclusions", "limitations", "endorsements"],
    "signature": ["signature", "authorized", "date signed"],
}

def _guess_section(text: str) -> str:
    lower = text.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return section
    return "body"
