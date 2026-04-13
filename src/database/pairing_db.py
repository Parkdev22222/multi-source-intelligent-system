"""
Pairing DB operations – insert and query temporal object pairs.
"""

import logging
from datetime import datetime
from typing import List, Optional, Set

from sqlalchemy import and_, func, or_
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


def update_pairings_detection_refs(old_det_ids: Set[str], new_det_id: Optional[str]) -> int:
    """탐지 결과 교체 후, 구 detection_id를 참조하던 PairingRecord를 새 ID로 업데이트.

    replace_detections_for_image() 호출 시 기존 DetectionRecord가 삭제되면
    PairingRecord의 current/past_detection_id가 stale해져 api_report_images의
    capture_time 폴백이 발동되고 다른 보고서 이미지가 노출되는 버그를 방지한다.
    """
    if not old_det_ids:
        return 0
    engine = get_engine()
    updated = 0
    with Session(engine) as session:
        pairings = session.query(PairingRecord).filter(
            (PairingRecord.current_detection_id.in_(old_det_ids)) |
            (PairingRecord.past_detection_id.in_(old_det_ids))
        ).all()
        for p in pairings:
            if p.current_detection_id in old_det_ids:
                p.current_detection_id = new_det_id
                updated += 1
            if p.past_detection_id in old_det_ids:
                p.past_detection_id = new_det_id
                updated += 1
        session.commit()
    logger.info(f"[PairingDB] Updated {updated} pairing detection refs after image edit")
    return updated


def delete_pairings_by_session(session_id: str) -> int:
    """세션의 모든 pairing_records를 삭제한다. 보고서 재생성 전 호출."""
    engine = get_engine()
    with Session(engine) as session:
        deleted = (
            session.query(PairingRecord)
            .filter(PairingRecord.session_id == session_id)
            .delete(synchronize_session=False)
        )
        session.commit()
    logger.info(f"[PairingDB] Deleted {deleted} pairings for session {session_id}")
    return deleted


def delete_older_duplicates_for_generated_pairings(
    generated_pairings: List[PairingRecord],
    *,
    session_id: str,
    source_type: str,
) -> int:
    """Delete older duplicate rows for a soon-to-be-inserted pairing batch.

    Newly generated rows are not yet inserted, so this only removes historical
    duplicates from DB. Duplicate 조건:
      - same session/source/status
      - and same current_detection_id or same past_detection_id
      - if both ids are missing, same class + rounded (6dp) lat/lon signature
    """
    if not generated_pairings:
        return 0

    engine = get_engine()
    deleted_total = 0
    with Session(engine) as session:
        for pr in generated_pairings:
            base_filters = [
                PairingRecord.session_id == session_id,
                PairingRecord.source_type == source_type,
                PairingRecord.status == pr.status,
            ]
            dup_filters = []

            if pr.current_detection_id:
                dup_filters.append(PairingRecord.current_detection_id == pr.current_detection_id)
            if pr.past_detection_id:
                dup_filters.append(PairingRecord.past_detection_id == pr.past_detection_id)

            if not dup_filters:
                cur_lat = round(pr.current_lat, 6) if pr.current_lat is not None else None
                cur_lon = round(pr.current_lon, 6) if pr.current_lon is not None else None
                past_lat = round(pr.past_lat, 6) if pr.past_lat is not None else None
                past_lon = round(pr.past_lon, 6) if pr.past_lon is not None else None

                dup_filters.append(and_(
                    PairingRecord.current_object_class == pr.current_object_class,
                    PairingRecord.past_object_class == pr.past_object_class,
                    func.round(PairingRecord.current_lat, 6) == cur_lat,
                    func.round(PairingRecord.current_lon, 6) == cur_lon,
                    func.round(PairingRecord.past_lat, 6) == past_lat,
                    func.round(PairingRecord.past_lon, 6) == past_lon,
                ))

            deleted_total += (
                session.query(PairingRecord)
                .filter(*base_filters, or_(*dup_filters))
                .delete(synchronize_session=False)
            )
        session.commit()

    if deleted_total:
        logger.info(
            "[PairingDB] Deleted %d historical duplicate pairings "
            "(session_id=%s, source_type=%s).",
            deleted_total, session_id, source_type,
        )
    return deleted_total


def get_session_location(session_id: str):
    """Return (lat_center, lon_center) for the first pairing of a session, or (None, None)."""
    engine = get_engine()
    with Session(engine) as session:
        row = (
            session.query(PairingRecord.lat_center, PairingRecord.lon_center)
            .filter(
                PairingRecord.session_id == session_id,
                PairingRecord.lat_center.isnot(None),
            )
            .first()
        )
        if row:
            return row[0], row[1]
        return None, None


def get_pairings_near_time(dt: datetime, window_seconds: int = 120) -> List[PairingRecord]:
    """시각 dt 기준으로 가장 가까운 pairing_time 배치를 반환한다.

    session_id가 없거나 session 기반 조회가 실패할 때의 폴백으로 사용.
    dt 이전 window_seconds 이내의 배치 중 가장 최신 pairing_time을 찾아 해당 배치 전체를 반환.
    """
    from datetime import timedelta
    engine = get_engine()
    from sqlalchemy import func
    with Session(engine) as session:
        lower = dt - timedelta(seconds=window_seconds)
        latest_pt = (
            session.query(func.max(PairingRecord.pairing_time))
            .filter(PairingRecord.pairing_time <= dt,
                    PairingRecord.pairing_time >= lower)
            .scalar()
        )
        if latest_pt is None:
            # 범위 내 없으면 dt 이전의 가장 최근 배치
            latest_pt = (
                session.query(func.max(PairingRecord.pairing_time))
                .filter(PairingRecord.pairing_time <= dt)
                .scalar()
            )
        if latest_pt is None:
            return []
        records = (
            session.query(PairingRecord)
            .filter(PairingRecord.pairing_time == latest_pt)
            .limit(500)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


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
