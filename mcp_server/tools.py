from typing import Any, Dict, List

from database import (
    Session,
    read_last_n_turns,
    read_milestones,
    read_preferences,
    write_conversation,
    write_milestone,
    write_preference,
)
from memory_schemas import Conversation, Milestone, UserPreferences
from vector_store import embed_and_store, semantic_search


TOOL_REGISTRY = [
    {
        "name": "memory_write",
        "description": (
            "Persist a memory object to the database. Accepts memory_type of "
            "'conversation', 'preference', or 'milestone'. Validates input against "
            "the corresponding Pydantic schema and stores structured data in SQLite. "
            "For conversation entries, content is also embedded and stored in the "
            "vector database for future semantic retrieval."
        ),
    },
    {
        "name": "memory_read",
        "description": (
            "Retrieve structured records from the SQLite database for a given user. "
            "Supports query_type values: 'last_n_turns' (returns the most recent N "
            "conversation turns), 'preferences' (returns stored user preferences), "
            "and 'milestones' (returns all academic milestones for the user)."
        ),
    },
    {
        "name": "memory_retrieve_by_context",
        "description": (
            "Perform a semantic similarity search over the vector database. Takes a "
            "natural language query, embeds it using the configured sentence-transformer "
            "model, and returns the top-k most contextually relevant stored memories "
            "for the specified user. Returns content, metadata, and similarity score."
        ),
    },
]


def execute_memory_write(
    db: Session, memory_type: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    if memory_type == "conversation":
        validated = Conversation(**data)
        memory_id = write_conversation(db, validated.model_dump())
        embed_and_store(
            document_id=memory_id,
            text=validated.content,
            metadata={
                "user_id": validated.user_id,
                "turn_id": validated.turn_id,
                "role": validated.role,
                "timestamp": validated.timestamp.isoformat(),
            },
        )
        return {"status": "success", "memory_id": memory_id}

    elif memory_type == "preference":
        validated = UserPreferences(**data)
        memory_id = write_preference(db, validated.model_dump())
        return {"status": "success", "memory_id": memory_id}

    elif memory_type in ("milestone", "milestones"):
        validated = Milestone(**data)
        memory_id = write_milestone(db, validated.model_dump())
        return {"status": "success", "memory_id": memory_id}

    else:
        raise ValueError(
            f"Unknown memory_type '{memory_type}'. "
            "Must be one of: 'conversation', 'preference', 'milestone'."
        )


def execute_memory_read(
    db: Session, user_id: str, query_type: str, params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if query_type == "last_n_turns":
        n = int(params.get("n", 10))
        return read_last_n_turns(db, user_id, n)

    elif query_type == "preferences":
        result = read_preferences(db, user_id)
        return [result] if result else []

    elif query_type == "milestones":
        return read_milestones(db, user_id)

    else:
        raise ValueError(
            f"Unknown query_type '{query_type}'. "
            "Must be one of: 'last_n_turns', 'preferences', 'milestones'."
        )


def execute_memory_retrieve_by_context(
    user_id: str, query_text: str, top_k: int
) -> List[Dict[str, Any]]:
    return semantic_search(query_text=query_text, user_id=user_id, top_k=top_k)
