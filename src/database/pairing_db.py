"""
Pairing DB operations – insert and query temporal object pairs.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import PairingRecord, create_pairing_engine
from src.config import PAIRING_DB_PATH

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_pairing_engine(PAIRING_DB_PATH)
    return _engine


def insert_pairing(pairing: PairingRecord) -> str:
    engine = get_engine()
    with Session(engine) as session:
        session.add(pairing)
        session.commit()
        session.refresh(pairing)
        logger.debug(f"[PairingDB] Inserted pairing id={pairing.id} status={pairing.status}")
        return pairing.id


def insert_pairings_bulk(pairings: List[PairingRecord]) -> List[str]:
    engine = get_engine()
    ids = []
    with Session(engine) as session:
        for p in pairings:
            session.add(p)
        session.commit()
        for p in pairings:
            session.refresh(p)
            ids.append(p.id)
    logger.info(f"[PairingDB] Bulk inserted {len(ids)} pairings.")
    return ids


def get_latest_pairings(session_id: Optional[str] = None, limit: int = 500) -> List[PairingRecord]:
    """
    Return the most recent set of pairing records.
    If session_id is provided, filter to that session.
    Otherwise returns the single most recent pairing_time batch.
    """
    engine = get_engine()
    with Session(engine) as session:
        query = session.query(PairingRecord)

        if session_id:
            query = query.filter(PairingRecord.session_id == session_id)
        else:
            from sqlalchemy import func
            latest_time = session.query(func.max(PairingRecord.pairing_time)).scalar()
            if latest_time is None:
                return []
            query = query.filter(PairingRecord.pairing_time == latest_time)

        records = query.limit(limit).all()
        for r in records:
            session.expunge(r)

        logger.info(f"[PairingDB] Retrieved {len(records)} pairing records.")
        return records


def get_pairings_by_session(session_id: str) -> List[PairingRecord]:
    return get_latest_pairings(session_id=session_id)


def get_session_ids_near(
    lat: float,
    lon: float,
    radius_deg: float = 0.1,
    limit: int = 20,
) -> List[str]:
    """Return distinct session_ids of pairings within radius_deg of (lat, lon)."""
    engine = get_engine()
    with Session(engine) as session:
        rows = (
            session.query(PairingRecord.session_id)
            .filter(
                PairingRecord.lat_center.between(lat - radius_deg, lat + radius_deg),
                PairingRecord.lon_center.between(lon - radius_deg, lon + radius_deg),
                PairingRecord.session_id.isnot(None),
            )
            .distinct()
            .limit(limit)
            .all()
        )
        return [r[0] for r in rows]
