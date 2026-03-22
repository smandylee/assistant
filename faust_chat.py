from __future__ import annotations

import argparse
import json
import os
import re
import sys

from google import genai

from emotion_engine import EmotionEngine
from faust_env import setup_env
from faust_memory import GeminiEmbedder, LocalMemoryStore
from faust_persona import INSTRUCTIONS
from faust_tools import current_datetime, web_search


def _extract_search_query(text: str) -> str:
    q = text.strip()
    q = re.sub(r"^\s*파우스트[, ]*", "", q)
    q = re.sub(r"(검색해줘|검색해 줘|검색해|찾아줘|찾아 줘|찾아|알려줘|알려 줘)\s*$", "", q)
    q = re.sub(r"^\s*(검색|뉴스|실시간)\s*", "", q)
    return q.strip(" ?!.")


def _run_tools_for_query(user_text: str) -> list[str]:
    text = user_text.strip()
    lower = text.lower()
    results: list[str] = []

    ask_time = any(k in text for k in ["몇 시", "몇시", "현재 시간", "지금 시간", "날짜", "오늘 날짜"])
    if ask_time:
        results.append(f"[도구:current_datetime]\n{current_datetime(9)}")

    ask_search = any(k in text for k in ["검색", "찾아", "뉴스", "실시간"]) or "latest" in lower
    if ask_search:
        query = _extract_search_query(text)
        if len(query) < 2:
            query = text
        results.append(f"[도구:web_search]\n{web_search(query, max_results=4)}")

    return results


def _build_prompt(
    user_text: str,
    memory_store: LocalMemoryStore,
    tool_results: list[str],
    emotion: dict[str, float | str] | None = None,
) -> str:
    memories = memory_store.search(user_text, top_k=3, min_score=0.1)
    memory_lines: list[str] = []
    for m in memories:
        role = m.get("role", "memo")
        text = str(m.get("text", "")).strip()
        if not text:
            continue
        # Prevent assistant-style canned phrases from contaminating next answers.
        if role == "assistant" and "파우스트의 기록 장치에는" in text:
            continue
        memory_lines.append(f"[과거 기억][{role}] {text}")

    memory_block = "\n".join(memory_lines) if memory_lines else "(관련 기억 없음)"
    tools_block = "\n\n".join(tool_results) if tool_results else "(호출된 도구 없음)"
    emotion = emotion or {"label": "analytical", "intensity": 0.35}
    emotion_label = str(emotion.get("label", "analytical"))
    emotion_style = EmotionEngine.style_instruction(emotion_label)
    return (
        f"{INSTRUCTIONS}\n\n"
        "중요 규칙:\n"
        "- 일반 질문에는 바로 핵심 답부터 말하십시오.\n"
        "- '파우스트의 기록 장치에는 ...' 문구는 사용자가 기억/과거 대화를 직접 물을 때만 사용하십시오.\n"
        "- 같은 문장을 반복하지 마십시오.\n\n"
        "도구 사용 지침:\n"
        "- 시간/날짜/최신 정보가 필요하면 도구를 우선 검토하십시오.\n"
        "- 도구를 호출했다면 결과를 근거로 답하십시오.\n\n"
        "현재 감정 상태:\n"
        f"- label={emotion_label}, intensity={emotion.get('intensity', 0.35)}\n"
        f"- 표현 지침: {emotion_style}\n\n"
        "도구 호출 결과(있으면 우선 신뢰):\n"
        f"{tools_block}\n\n"
        "아래는 관련 과거 대화 메모입니다. 모호하면 추측하지 마십시오.\n"
        f"{memory_block}\n\n"
        f"[사용자 질문]\n{user_text}\n"
    )


def _extract_response_text(resp: object) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()
    return "파우스트는 현재 응답을 생성할 수 없습니다."


def _build_genai_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gcp-key.json")
        if os.path.exists(cred_path):
            try:
                with open(cred_path, "r", encoding="utf-8") as f:
                    project = json.load(f).get("project_id")
            except Exception:
                project = None

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "Vertex AI 사용을 위해 GOOGLE_CLOUD_PROJECT가 필요합니다. "
            "(.env에 설정하거나 gcp-key.json에 project_id가 있어야 합니다)"
        )

    return genai.Client(vertexai=True, project=project, location=location)


def main() -> int:
    setup_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="")
    args = parser.parse_args()

    user_text = (args.message or "").strip()
    if not user_text:
        user_text = sys.stdin.read().strip()
    if not user_text:
        print("질문이 비어 있습니다.", flush=True)
        return 1

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    memory_file = os.getenv("MEMORY_FILE", "memory/memory.jsonl")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    memory_store = LocalMemoryStore(memory_file, max_scan_rows=300, embedder=GeminiEmbedder(embed_model))
    emotion_engine = EmotionEngine()
    memory_store.add("user", user_text)

    tool_results = _run_tools_for_query(user_text)
    emotion = emotion_engine.update(user_text).as_dict()
    prompt = _build_prompt(user_text, memory_store, tool_results, emotion=emotion)
    try:
        client = _build_genai_client()
    except Exception as exc:
        print(f"Gemini 클라이언트 초기화 실패: {exc}", flush=True)
        return 1

    try:
        resp = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as exc:
        print(f"응답 생성 실패: {exc}", flush=True)
        return 1

    answer = _extract_response_text(resp)
    memory_store.add("assistant", answer)
    print(answer, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
