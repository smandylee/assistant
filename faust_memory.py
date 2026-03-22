from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from google import genai
from livekit.agents import Agent


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts).strip()
    return str(content)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[0-9A-Za-z가-힣]+", text)}


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class GeminiEmbedder:
    def __init__(self, model: str):
        self.model = model
        self.client: genai.Client | None = None
        self.error: str | None = None
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                return

            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self.client = genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:
            self.error = str(exc)

    def embed(self, text: str) -> list[float] | None:
        if not self.client:
            return None
        text = text.strip()
        if not text:
            return None
        try:
            res = self.client.models.embed_content(model=self.model, contents=text)
            embeddings = getattr(res, "embeddings", None) or []
            if not embeddings:
                return None
            values = getattr(embeddings[0], "values", None) or []
            return [float(v) for v in values]
        except Exception:
            return None


class LocalMemoryStore:
    def __init__(
        self,
        file_path: str,
        max_scan_rows: int = 300,
        embedder: GeminiEmbedder | None = None,
    ):
        self.path = Path(file_path)
        self.max_scan_rows = max_scan_rows
        self.embedder = embedder
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, role: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        row: dict[str, Any] = {"role": role, "text": text, "ts": time.time()}
        embedding = self.embedder.embed(text) if self.embedder else None
        if embedding:
            row["embedding"] = embedding
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_recent(self) -> list[dict]:
        rows: deque[dict] = deque(maxlen=max(1, self.max_scan_rows))
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(rows)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.1) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        query_embedding = self.embedder.embed(query) if self.embedder else None
        now = time.time()
        scored: list[tuple[float, dict]] = []
        for row in self._read_recent():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            d_tokens = _tokenize(text)
            if not d_tokens:
                continue

            overlap = len(q_tokens & d_tokens) / math.sqrt(len(q_tokens) * len(d_tokens))
            semantic_score = 0.0
            if query_embedding:
                row_embedding = row.get("embedding")
                if isinstance(row_embedding, list) and row_embedding:
                    try:
                        semantic_score = _cosine_similarity(
                            query_embedding, [float(v) for v in row_embedding]
                        )
                    except (TypeError, ValueError):
                        semantic_score = 0.0

            age_h = max(0.0, (now - float(row.get("ts", now))) / 3600.0)
            recency_bonus = 0.05 / (1.0 + age_h)
            score = (0.75 * semantic_score if query_embedding else 0.0) + (0.25 * overlap) + recency_bonus
            if score >= min_score:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:top_k]]


class FaustAgent(Agent):
    def __init__(
        self,
        memory_store: LocalMemoryStore,
        memory_top_k: int,
        memory_min_score: float,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory_store = memory_store
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        user_text = extract_text(new_message.content).strip()
        if not user_text:
            return

        self.memory_store.add("user", user_text)
        memories = self.memory_store.search(
            user_text, top_k=self.memory_top_k, min_score=self.memory_min_score
        )
        if not memories:
            return

        memory_lines = []
        for m in memories:
            role = m.get("role", "memo")
            text = str(m.get("text", "")).strip()
            if text:
                memory_lines.append(f"[과거 기억][{role}] {text}")

        if not memory_lines:
            return

        turn_ctx.add_message(
            role="system",
            content=(
                "아래는 현재 질문과 관련된 과거 대화 메모입니다. "
                "사실 확인용으로만 참고하고, 없거나 모호하면 추측하지 마세요.\n"
                + "\n".join(memory_lines)
            ),
        )
