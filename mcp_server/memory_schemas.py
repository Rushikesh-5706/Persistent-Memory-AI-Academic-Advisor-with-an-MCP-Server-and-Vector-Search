from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Dict, Optional


class Conversation(BaseModel):
    user_id: str
    turn_id: int
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserPreferences(BaseModel):
    user_id: str
    preferences: Dict[str, Any]


class Milestone(BaseModel):
    user_id: str
    milestone_id: str
    description: str
    status: str
    date_achieved: Optional[datetime] = None


class MemoryWriteRequest(BaseModel):
    memory_type: str
    data: Dict[str, Any]


class MemoryReadRequest(BaseModel):
    user_id: str
    query_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MemoryRetrieveRequest(BaseModel):
    user_id: str
    query_text: str
    top_k: int = 5
