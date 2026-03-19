"""
Academic Advisor Agent

A conversational AI academic advisor that uses the MCP server for persistent
memory. The agent retrieves relevant context before each response and writes
each conversation turn to long-term storage after responding.

Usage:
    Set OLLAMA_BASE_URL and OLLAMA_MODEL in your .env file, then run:
    python agent/agent.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """You are a knowledgeable and supportive academic advisor. Your role is to help students
plan their academic journey, choose courses, set goals, and overcome challenges. You have access to
persistent memory of past conversations with each student. When relevant context from previous
interactions is provided to you, use it to give personalized and consistent guidance. Be concise,
specific, and encouraging."""


def wait_for_server(max_retries: int = 10, delay: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting for MCP server... ({attempt + 1}/{max_retries})")
        time.sleep(delay)
    print("MCP server did not become available. Exiting.")
    sys.exit(1)


def memory_write(memory_type: str, data: Dict[str, Any]) -> Optional[str]:
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/invoke/memory_write",
            json={"memory_type": memory_type, "data": data},
            timeout=10,
        )
        if response.status_code == 201:
            return response.json().get("memory_id")
    except requests.exceptions.RequestException as exc:
        print(f"[WARNING] memory_write failed: {exc}")
    return None


def memory_read(user_id: str, query_type: str, params: Dict[str, Any]) -> List[Dict]:
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/invoke/memory_read",
            json={"user_id": user_id, "query_type": query_type, "params": params},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("results", [])
    except requests.exceptions.RequestException as exc:
        print(f"[WARNING] memory_read failed: {exc}")
    return []


def memory_retrieve_by_context(
    user_id: str, query_text: str, top_k: int = 3
) -> List[Dict]:
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/invoke/memory_retrieve_by_context",
            json={"user_id": user_id, "query_text": query_text, "top_k": top_k},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("results", [])
    except requests.exceptions.RequestException as exc:
        print(f"[WARNING] memory_retrieve_by_context failed: {exc}")
    return []


def call_ollama(messages: List[Dict[str, str]]) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as exc:
        return f"[ERROR] Could not reach Ollama: {exc}"
    except (KeyError, json.JSONDecodeError) as exc:
        return f"[ERROR] Unexpected response from Ollama: {exc}"


def build_context_block(
    recent_turns: List[Dict], semantic_results: List[Dict]
) -> str:
    parts = []

    if recent_turns:
        parts.append("Recent conversation history:")
        for turn in recent_turns[-5:]:
            parts.append(f"  [{turn['role']}]: {turn['content']}")

    if semantic_results:
        parts.append("\nRelevant past context (from memory):")
        for result in semantic_results[:2]:
            score = result.get("score", 0)
            if score > 0.3:
                parts.append(f"  - {result['content']} (relevance: {score:.2f})")

    return "\n".join(parts) if parts else ""


def run_advisor_session(user_id: str) -> None:
    print(f"\nAcademic Advisor — session started for user: {user_id}")
    print("Type 'quit' or 'exit' to end the session.\n")

    turn_counter_response = memory_read(
        user_id=user_id,
        query_type="last_n_turns",
        params={"n": 1},
    )
    next_turn_id = 1
    if turn_counter_response:
        next_turn_id = turn_counter_response[-1]["turn_id"] + 1

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        recent_turns = memory_read(
            user_id=user_id,
            query_type="last_n_turns",
            params={"n": 10},
        )
        semantic_results = memory_retrieve_by_context(
            user_id=user_id,
            query_text=user_input,
            top_k=3,
        )

        context_block = build_context_block(recent_turns, semantic_results)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_block:
            messages.append(
                {
                    "role": "system",
                    "content": f"Context from memory:\n{context_block}",
                }
            )
        messages.append({"role": "user", "content": user_input})

        response_text = call_ollama(messages)
        print(f"\nAdvisor: {response_text}\n")

        memory_write(
            memory_type="conversation",
            data={
                "user_id": user_id,
                "turn_id": next_turn_id,
                "role": "user",
                "content": user_input,
            },
        )
        next_turn_id += 1

        memory_write(
            memory_type="conversation",
            data={
                "user_id": user_id,
                "turn_id": next_turn_id,
                "role": "assistant",
                "content": response_text,
            },
        )
        next_turn_id += 1


if __name__ == "__main__":
    wait_for_server()
    user_id = input("Enter your student ID: ").strip() or "student_001"
    run_advisor_session(user_id)
