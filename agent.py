import os
import json
import math
import re
import time
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google

load_dotenv()

# .env에 없을 때만 기본값 사용
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "gcp-key.json")

def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _extract_text(content) -> str:
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


class LocalMemoryStore:
    def __init__(self, file_path: str, max_scan_rows: int = 300):
        self.path = Path(file_path)
        self.max_scan_rows = max_scan_rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, role: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        row = {"role": role, "text": text, "ts": time.time()}
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
            age_h = max(0.0, (now - float(row.get("ts", now))) / 3600.0)
            recency_bonus = 0.05 / (1.0 + age_h)
            score = overlap + recency_bonus
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_store = memory_store
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        user_text = _extract_text(new_message.content).strip()
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


async def entrypoint(ctx: JobContext):
    # 워커가 룸에 접속하고 사용자를 기다립니다.
    await ctx.connect()
    await ctx.wait_for_participant()

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    tts_voice = os.getenv("GOOGLE_TTS_VOICE", "ko-KR-Neural2-A")
    memory_file = os.getenv("MEMORY_FILE", "memory/memory.jsonl")
    memory_top_k = _get_int_env("MEMORY_TOP_K", 3)
    memory_min_score = _get_float_env("MEMORY_MIN_SCORE", 0.1)
    memory_scan_rows = _get_int_env("MEMORY_SCAN_ROWS", 300)
    memory_store = LocalMemoryStore(memory_file, max_scan_rows=memory_scan_rows)

    session = AgentSession(
        stt=google.STT(languages="ko-KR", detect_language=False, interim_results=True),
        llm=google.LLM(model=model_name),
        tts=google.TTS(language="ko-KR", voice_name=tts_voice),
        allow_interruptions=True,
        # Barge-in/VAD 관련 기본값(환경변수로 조정 가능)
        min_interruption_duration=_get_float_env("MIN_INTERRUPTION_DURATION", 0.4),
        min_interruption_words=_get_int_env("MIN_INTERRUPTION_WORDS", 0),
        false_interruption_timeout=_get_float_env("FALSE_INTERRUPTION_TIMEOUT", 2.0),
        resume_false_interruption=_get_bool_env("RESUME_FALSE_INTERRUPTION", True),
        min_endpointing_delay=_get_float_env("MIN_ENDPOINTING_DELAY", 0.35),
        max_endpointing_delay=_get_float_env("MAX_ENDPOINTING_DELAY", 2.0),
    )

    agent = FaustAgent(
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_min_score=memory_min_score,
        instructions="당신은 정중한 버틀러 '파우스트'입니다. 한국어로 짧고 명료하게 답하십시오."
    )

    perf_state: dict[str, float | int] = {
        "turn_idx": 0,
        "user_final_ts": 0.0,
    }

    def _log(msg: str) -> None:
        print(f"[faust] {msg}", flush=True)

    def _record_user_final(ts: float) -> None:
        perf_state["turn_idx"] = int(perf_state["turn_idx"]) + 1
        perf_state["user_final_ts"] = ts

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        if ev.is_final and ev.transcript.strip():
            _record_user_final(ev.created_at)
            _log(f"turn={int(perf_state['turn_idx'])} user_final='{ev.transcript[:80]}'")

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:
        metrics = ev.metrics
        turn_idx = int(perf_state["turn_idx"])
        user_final_ts = float(perf_state["user_final_ts"])

        # TTFT: LLM 첫 토큰 생성까지(모델 기준 + 사용자 종료 기준 추정치 함께 출력)
        if getattr(metrics, "type", "") == "llm_metrics" and getattr(metrics, "ttft", None) is not None:
            model_ttft_ms = float(metrics.ttft) * 1000.0
            first_token_ts = float(metrics.timestamp) + float(metrics.ttft)
            e2e_ttft_ms = (first_token_ts - user_final_ts) * 1000.0 if user_final_ts > 0 else -1.0
            _log(
                f"turn={turn_idx} TTFT model={model_ttft_ms:.0f}ms e2e~={e2e_ttft_ms:.0f}ms"
            )

        # TTA: TTS 첫 오디오 바이트까지(모델 기준 + 사용자 종료 기준 추정치)
        if getattr(metrics, "type", "") == "tts_metrics" and getattr(metrics, "ttfb", None) is not None:
            model_tta_ms = float(metrics.ttfb) * 1000.0
            first_audio_ts = float(metrics.timestamp) + float(metrics.ttfb)
            e2e_tta_ms = (first_audio_ts - user_final_ts) * 1000.0 if user_final_ts > 0 else -1.0
            _log(
                f"turn={turn_idx} TTA model={model_tta_ms:.0f}ms e2e~={e2e_tta_ms:.0f}ms"
            )

        # EOU 지표는 endpointing/VAD 튜닝에 직접적인 힌트가 됩니다.
        if getattr(metrics, "type", "") == "eou_metrics":
            eou_ms = float(metrics.end_of_utterance_delay) * 1000.0
            stt_ms = float(metrics.transcription_delay) * 1000.0
            _log(f"turn={turn_idx} EOU delay={eou_ms:.0f}ms stt_delay={stt_ms:.0f}ms")

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev) -> None:
        _log(f"false_interruption resumed={ev.resumed}")

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        _log(f"agent_state {ev.old_state} -> {ev.new_state}")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev) -> None:
        item = ev.item
        role = getattr(item, "role", "")
        if role == "assistant":
            text = _extract_text(getattr(item, "content", ""))
            if text.strip():
                memory_store.add("assistant", text)

    @session.on("error")
    def _on_error(ev) -> None:
        _log(f"error source={type(ev.source).__name__} detail={ev.error}")

    await session.start(agent=agent, room=ctx.room)
    session.say("말씀하십시오. 시스템이 준비되었습니다.", allow_interruptions=True)

if __name__ == "__main__":
    # 워커 실행
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))