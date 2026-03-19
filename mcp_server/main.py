from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from database import (
    SessionLocal,
    init_db,
)
from memory_schemas import MemoryReadRequest, MemoryRetrieveRequest, MemoryWriteRequest
from tools import (
    TOOL_REGISTRY,
    execute_memory_read,
    execute_memory_retrieve_by_context,
    execute_memory_write,
)
from vector_store import get_collection, get_embedding_model, get_vector_count

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_embedding_model()
    get_collection()
    yield


app = FastAPI(
    title="Academic Advisor MCP Server",
    description=(
        "Memory, Control, and Process server providing persistent memory tools "
        "for an AI academic advisor agent. Exposes structured (SQLite) and "
        "semantic (ChromaDB) memory operations via a RESTful interface."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tools", status_code=status.HTTP_200_OK)
def list_tools() -> Dict[str, List[Dict[str, str]]]:
    return {"tools": TOOL_REGISTRY}


@app.post("/invoke/memory_write", status_code=status.HTTP_201_CREATED)
def invoke_memory_write(request: MemoryWriteRequest) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        result = execute_memory_write(
            db=db,
            memory_type=request.memory_type,
            data=request.data,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory write failed: {str(exc)}",
        )
    finally:
        db.close()


@app.post("/invoke/memory_read", status_code=status.HTTP_200_OK)
def invoke_memory_read(request: MemoryReadRequest) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        results = execute_memory_read(
            db=db,
            user_id=request.user_id,
            query_type=request.query_type,
            params=request.params,
        )
        return {"results": results}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory read failed: {str(exc)}",
        )
    finally:
        db.close()


@app.post("/invoke/memory_retrieve_by_context", status_code=status.HTTP_200_OK)
def invoke_memory_retrieve_by_context(
    request: MemoryRetrieveRequest,
) -> Dict[str, Any]:
    try:
        results = execute_memory_retrieve_by_context(
            user_id=request.user_id,
            query_text=request.query_text,
            top_k=request.top_k,
        )
        return {"results": results}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(exc)}",
        )


@app.get("/debug/vector_count", status_code=status.HTTP_200_OK)
def debug_vector_count() -> Dict[str, int]:
    return {"count": get_vector_count()}
