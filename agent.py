import os
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


async def entrypoint(ctx: JobContext):
    # 워커가 룸에 접속하고 사용자를 기다립니다.
    await ctx.connect()
    await ctx.wait_for_participant()

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    tts_voice = os.getenv("GOOGLE_TTS_VOICE", "ko-KR-Neural2-A")

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

    agent = Agent(
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

    @session.on("error")
    def _on_error(ev) -> None:
        _log(f"error source={type(ev.source).__name__} detail={ev.error}")

    await session.start(agent=agent, room=ctx.room)
    session.say("말씀하십시오. 시스템이 준비되었습니다.", allow_interruptions=True)

if __name__ == "__main__":
    # 워커 실행
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))