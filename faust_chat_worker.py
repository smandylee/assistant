from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from typing import Any

from google.genai import types

from emotion_engine import EmotionEngine
from faust_chat import (
    _build_genai_client,
    _build_prompt,
    _extract_response_text,
)
from faust_env import setup_env
from faust_memory import GeminiEmbedder, LocalMemoryStore
from faust_tools import current_datetime, web_search


def _write(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _tool_web_search(query: str, max_results: int = 4) -> str:
    return web_search(query=query, max_results=max_results)


def _tool_current_datetime(tz_offset_hours: int = 9) -> str:
    return current_datetime(tz_offset_hours=tz_offset_hours)


def _generate_reply(
    *,
    client: Any,
    model_name: str,
    prompt: str,
) -> str:
    cfg = types.GenerateContentConfig(
        tools=[_tool_web_search, _tool_current_datetime],
        automaticFunctionCalling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximumRemoteCalls=3,
        ),
        maxOutputTokens=700,
        temperature=0.6,
    )
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg,
    )
    return _extract_response_text(resp)


def _build_proactive_message(emotion: dict[str, float | str]) -> str:
    label = str(emotion.get("label", "analytical"))
    if label in {"urgent", "frustrated", "angry"}:
        return "파우스트는 현재 우선순위 점검을 권고합니다. 진행 중인 작업의 상태를 보고하십시오."
    if label in {"playful", "cold_amused"}:
        return "파우스트는 잠깐의 점검 시간을 제안합니다. 다음으로 진행할 항목을 지정하십시오."
    if label in {"calm", "analytical"}:
        return "파우스트는 대기 중입니다. 필요하다면 다음 지시를 입력하십시오."
    return "파우스트가 준비되었습니다. 다음 요청을 입력하십시오."


async def main_async() -> int:
    setup_env()

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    memory_file = os.getenv("MEMORY_FILE", "memory/memory.jsonl")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    proactive_idle_seconds = float(os.getenv("PROACTIVE_IDLE_SECONDS", "120"))
    proactive_cooldown_seconds = float(os.getenv("PROACTIVE_COOLDOWN_SECONDS", "90"))
    memory_store = LocalMemoryStore(memory_file, max_scan_rows=300, embedder=GeminiEmbedder(embed_model))
    emotion_engine = EmotionEngine()
    last_user_ts = time.monotonic()
    last_proactive_ts = 0.0

    try:
        client = _build_genai_client()
    except Exception as exc:
        _write({"type": "fatal", "error": f"Gemini 클라이언트 초기화 실패: {exc}"})
        return 1

    _write({"type": "ready"})

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _stdin_reader() -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":
                await queue.put(None)
                return
            await queue.put(line.rstrip("\r\n"))

    async def _proactive_loop() -> None:
        nonlocal last_proactive_ts
        while True:
            await asyncio.sleep(2.0)
            now = time.monotonic()
            if now - last_user_ts < proactive_idle_seconds:
                continue
            if now - last_proactive_ts < proactive_cooldown_seconds:
                continue

            emotion = emotion_engine.state.as_dict()
            proactive = _build_proactive_message(emotion)
            memory_store.add("assistant", proactive)
            _write({"type": "proactive", "message": proactive, "emotion": emotion})
            last_proactive_ts = now

    stdin_task = asyncio.create_task(_stdin_reader())
    proactive_task = asyncio.create_task(_proactive_loop())
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            if not line.strip():
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                _write({"type": "error", "error": "invalid_json"})
                continue

            req_id = req.get("id")
            text = str(req.get("message", "")).strip()
            if not req_id:
                _write({"type": "error", "error": "missing_id"})
                continue
            if not text:
                _write({"type": "response", "id": req_id, "ok": False, "error": "empty_message"})
                continue

            last_user_ts = time.monotonic()
            try:
                memory_store.add("user", text)
                emotion = emotion_engine.update(text).as_dict()
                # Native function-calling will decide if/when tools are needed.
                prompt = _build_prompt(text, memory_store, [], emotion=emotion)
                answer = await asyncio.to_thread(
                    _generate_reply,
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                )
                memory_store.add("assistant", answer)
                _write(
                    {
                        "type": "response",
                        "id": req_id,
                        "ok": True,
                        "reply": answer,
                        "emotion": emotion,
                    }
                )
            except Exception as exc:
                _write({"type": "response", "id": req_id, "ok": False, "error": str(exc)})
    finally:
        stdin_task.cancel()
        proactive_task.cancel()
        with contextlib.suppress(Exception):
            await stdin_task
        with contextlib.suppress(Exception):
            await proactive_task

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
