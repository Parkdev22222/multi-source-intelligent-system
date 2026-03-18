"""
Reports DB operations – insert and query generated military intelligence reports.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import ReportRecord, create_report_engine
from src.config import REPORTS_DB_PATH

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_report_engine(REPORTS_DB_PATH)
    return _engine


def insert_report(
    report_time: datetime,
    report_content: str,
    llm_model: str,
    llm_backend: str,
    pairing_count: int,
    session_id: Optional[str] = None,
    file_path: Optional[str] = None,
) -> ReportRecord:
    """
    Save a generated report to the Reports DB.

    Args:
        report_time:    The timestamp embedded in the report header (LLM generation time).
        report_content: Full report text (including the header block).
        llm_model:      Model identifier used for generation (e.g. LGAI-EXAONE/EXAONE-4.0-32B-Instruct).
        llm_backend:    Backend used: "huggingface" | "ollama".
        pairing_count:  Number of pairing records that were analysed.
        session_id:     Pipeline session UUID (optional).
        file_path:      Absolute path to the saved .txt file on disk (optional).

    Returns:
        The inserted ReportRecord ORM object.
    """
    engine = get_engine()
    with Session(engine) as session:
        record = ReportRecord(
            report_time=report_time,
            report_content=report_content,
            llm_model=llm_model,
            llm_backend=llm_backend,
            pairing_count=pairing_count,
            session_id=session_id,
            file_path=file_path,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(
            f"[ReportsDB] Saved report id={record.id}  "
            f"saved_time={record.saved_time}  file_path={record.file_path}"
        )
        return record


def get_all_reports(limit: int = 100) -> List[ReportRecord]:
    """Return all reports ordered by saved_time descending."""
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(ReportRecord)
            .order_by(ReportRecord.saved_time.desc())
            .limit(limit)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


def get_report_by_id(report_id: str) -> Optional[ReportRecord]:
    engine = get_engine()
    with Session(engine) as session:
        record = session.get(ReportRecord, report_id)
        if record:
            session.expunge(record)
        return record


def get_reports_by_session(session_id: str) -> List[ReportRecord]:
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(ReportRecord)
            .filter(ReportRecord.session_id == session_id)
            .order_by(ReportRecord.saved_time.desc())
            .all()
        )
        for r in records:
            session.expunge(r)
        return records
