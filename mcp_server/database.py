import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "/app/data/advisor_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ConversationRecord(Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("user_id", "turn_id", name="uq_user_turn"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    turn_id = Column(Integer, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserPreferencesRecord(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    preferences = Column(Text, nullable=False)


class MilestoneRecord(Base):
    __tablename__ = "milestones"
    __table_args__ = (
        UniqueConstraint("user_id", "milestone_id", name="uq_user_milestone"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    milestone_id = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    date_achieved = Column(DateTime, nullable=True)


def init_db() -> None:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text(f"PRAGMA busy_timeout=5000"))
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def write_conversation(db: Session, data: Dict[str, Any]) -> str:
    from datetime import timezone

    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.utcnow()

    existing = (
        db.query(ConversationRecord)
        .filter_by(user_id=data["user_id"], turn_id=data["turn_id"])
        .first()
    )
    if existing:
        existing.role = data["role"]
        existing.content = data["content"]
        existing.timestamp = timestamp
        db.commit()
        memory_id = f"conv_{data['user_id']}_{data['turn_id']}"
        return memory_id

    record = ConversationRecord(
        user_id=data["user_id"],
        turn_id=data["turn_id"],
        role=data["role"],
        content=data["content"],
        timestamp=timestamp,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return f"conv_{data['user_id']}_{data['turn_id']}"


def write_preference(db: Session, data: Dict[str, Any]) -> str:
    prefs_json = json.dumps(data["preferences"])
    existing = (
        db.query(UserPreferencesRecord).filter_by(user_id=data["user_id"]).first()
    )
    if existing:
        existing.preferences = prefs_json
        db.commit()
    else:
        record = UserPreferencesRecord(
            user_id=data["user_id"], preferences=prefs_json
        )
        db.add(record)
        db.commit()
    return f"pref_{data['user_id']}"


def write_milestone(db: Session, data: Dict[str, Any]) -> str:
    date_achieved = data.get("date_achieved")
    if isinstance(date_achieved, str):
        date_achieved = datetime.fromisoformat(date_achieved.replace("Z", "+00:00"))

    existing = (
        db.query(MilestoneRecord)
        .filter_by(user_id=data["user_id"], milestone_id=data["milestone_id"])
        .first()
    )
    if existing:
        existing.description = data["description"]
        existing.status = data["status"]
        existing.date_achieved = date_achieved
        db.commit()
        return f"ms_{data['user_id']}_{data['milestone_id']}"

    record = MilestoneRecord(
        user_id=data["user_id"],
        milestone_id=data["milestone_id"],
        description=data["description"],
        status=data["status"],
        date_achieved=date_achieved,
    )
    db.add(record)
    db.commit()
    return f"ms_{data['user_id']}_{data['milestone_id']}"


def read_last_n_turns(db: Session, user_id: str, n: int) -> List[Dict[str, Any]]:
    records = (
        db.query(ConversationRecord)
        .filter_by(user_id=user_id)
        .order_by(ConversationRecord.timestamp.desc())
        .limit(n)
        .all()
    )
    result = []
    for r in reversed(records):
        result.append(
            {
                "user_id": r.user_id,
                "turn_id": r.turn_id,
                "role": r.role,
                "content": r.content,
                "timestamp": r.timestamp.isoformat(),
            }
        )
    return result


def read_preferences(db: Session, user_id: str) -> Optional[Dict[str, Any]]:
    record = db.query(UserPreferencesRecord).filter_by(user_id=user_id).first()
    if not record:
        return None
    return {"user_id": record.user_id, "preferences": json.loads(record.preferences)}


def read_milestones(db: Session, user_id: str) -> List[Dict[str, Any]]:
    records = db.query(MilestoneRecord).filter_by(user_id=user_id).all()
    result = []
    for r in records:
        result.append(
            {
                "user_id": r.user_id,
                "milestone_id": r.milestone_id,
                "description": r.description,
                "status": r.status,
                "date_achieved": r.date_achieved.isoformat() if r.date_achieved else None,
            }
        )
    return result
