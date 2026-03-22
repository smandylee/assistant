from __future__ import annotations

import os
from typing import Any

from livekit.agents import AgentSession, JobContext
from livekit.plugins import elevenlabs, google

from faust_env import get_bool_env, get_float_env, get_int_env, setup_env
from faust_memory import FaustAgent, GeminiEmbedder, LocalMemoryStore, extract_text
from faust_persona import INSTRUCTIONS
from faust_tools import TOOLS

try:
    from livekit.agents.voice.agent_session import VoiceActivityVideoSampler
except Exception:
    VoiceActivityVideoSampler = None


async def entrypoint(ctx: JobContext) -> None:
    setup_env()

    await ctx.connect()
    await ctx.wait_for_participant()

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    tts_voice = os.getenv("GOOGLE_TTS_VOICE", "ko-KR-Neural2-A")
    eleven_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "FKpLBDCIkrMlLHfQVK29")
    eleven_model = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
    eleven_language = os.getenv("ELEVENLABS_LANGUAGE", "ko")
    use_elevenlabs_tts = get_bool_env("USE_ELEVENLABS_TTS", True)
    eleven_api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")

    memory_file = os.getenv("MEMORY_FILE", "memory/memory.jsonl")
    memory_top_k = get_int_env("MEMORY_TOP_K", 3)
    memory_min_score = get_float_env("MEMORY_MIN_SCORE", 0.1)
    memory_scan_rows = get_int_env("MEMORY_SCAN_ROWS", 300)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

    embedder = GeminiEmbedder(embedding_model)
    memory_store = LocalMemoryStore(memory_file, max_scan_rows=memory_scan_rows, embedder=embedder)

    vision_speaking_fps = get_float_env("VISION_SPEAKING_FPS", 1.0)
    vision_silent_fps = get_float_env("VISION_SILENT_FPS", 0.3)
    video_sampler = None
    if VoiceActivityVideoSampler:
        video_sampler = VoiceActivityVideoSampler(
            speaking_fps=vision_speaking_fps,
            silent_fps=vision_silent_fps,
        )

    tts_engine: Any
    tts_provider = "google"
    if use_elevenlabs_tts and eleven_api_key:
        tts_engine = elevenlabs.TTS(
            api_key=eleven_api_key,
            voice_id=eleven_voice_id,
            model=eleven_model,
            language=eleven_language,
        )
        tts_provider = "elevenlabs"
    else:
        tts_engine = google.TTS(language="ko-KR", voice_name=tts_voice)

    session = AgentSession(
        stt=google.STT(languages="ko-KR", detect_language=False, interim_results=True),
        llm=google.LLM(model=model_name),
        tts=tts_engine,
        allow_interruptions=True,
        tools=TOOLS,
        video_sampler=video_sampler,
        min_interruption_duration=get_float_env("MIN_INTERRUPTION_DURATION", 0.4),
        min_interruption_words=get_int_env("MIN_INTERRUPTION_WORDS", 0),
        false_interruption_timeout=get_float_env("FALSE_INTERRUPTION_TIMEOUT", 2.0),
        resume_false_interruption=get_bool_env("RESUME_FALSE_INTERRUPTION", True),
        min_endpointing_delay=get_float_env("MIN_ENDPOINTING_DELAY", 0.35),
        max_endpointing_delay=get_float_env("MAX_ENDPOINTING_DELAY", 2.0),
    )

    agent = FaustAgent(
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_min_score=memory_min_score,
        instructions=INSTRUCTIONS,
    )

    perf_state: dict[str, float | int] = {"turn_idx": 0, "user_final_ts": 0.0}

    def _log(msg: str) -> None:
        print(f"[faust] {msg}", flush=True)

    if embedder.client:
        _log(f"semantic memory enabled (model={embedding_model})")
    else:
        _log(f"semantic memory disabled ({embedder.error or 'embed client unavailable'})")

    _log(
        "vision enabled "
        f"(speaking_fps={vision_speaking_fps}, silent_fps={vision_silent_fps}, sampler={'on' if video_sampler else 'off'})"
    )
    _log("function tools enabled (web_search, current_datetime)")
    if tts_provider == "elevenlabs":
        _log(f"tts provider=elevenlabs voice_id={eleven_voice_id} model={eleven_model}")
    else:
        _log("tts provider=google (set ELEVENLABS_API_KEY to enable ElevenLabs)")

    def _record_user_final(ts: float) -> None:
        perf_state["turn_idx"] = int(perf_state["turn_idx"]) + 1
        perf_state["user_final_ts"] = ts

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: Any) -> None:
        if ev.is_final and ev.transcript.strip():
            _record_user_final(ev.created_at)
            _log(f"turn={int(perf_state['turn_idx'])} user_final='{ev.transcript[:80]}'")

    @session.on("metrics_collected")
    def _on_metrics(ev: Any) -> None:
        metrics = ev.metrics
        turn_idx = int(perf_state["turn_idx"])
        user_final_ts = float(perf_state["user_final_ts"])

        if getattr(metrics, "type", "") == "llm_metrics" and getattr(metrics, "ttft", None) is not None:
            model_ttft_ms = float(metrics.ttft) * 1000.0
            first_token_ts = float(metrics.timestamp) + float(metrics.ttft)
            e2e_ttft_ms = (first_token_ts - user_final_ts) * 1000.0 if user_final_ts > 0 else -1.0
            _log(f"turn={turn_idx} TTFT model={model_ttft_ms:.0f}ms e2e~={e2e_ttft_ms:.0f}ms")

        if getattr(metrics, "type", "") == "tts_metrics" and getattr(metrics, "ttfb", None) is not None:
            model_tta_ms = float(metrics.ttfb) * 1000.0
            first_audio_ts = float(metrics.timestamp) + float(metrics.ttfb)
            e2e_tta_ms = (first_audio_ts - user_final_ts) * 1000.0 if user_final_ts > 0 else -1.0
            _log(f"turn={turn_idx} TTA model={model_tta_ms:.0f}ms e2e~={e2e_tta_ms:.0f}ms")

        if getattr(metrics, "type", "") == "eou_metrics":
            eou_ms = float(metrics.end_of_utterance_delay) * 1000.0
            stt_ms = float(metrics.transcription_delay) * 1000.0
            _log(f"turn={turn_idx} EOU delay={eou_ms:.0f}ms stt_delay={stt_ms:.0f}ms")

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev: Any) -> None:
        _log(f"false_interruption resumed={ev.resumed}")

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: Any) -> None:
        _log(f"agent_state {ev.old_state} -> {ev.new_state}")

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: Any) -> None:
        item = ev.item
        role = getattr(item, "role", "")
        if role == "assistant":
            text = extract_text(getattr(item, "content", ""))
            if text.strip():
                memory_store.add("assistant", text)

    @session.on("error")
    def _on_error(ev: Any) -> None:
        _log(f"error source={type(ev.source).__name__} detail={ev.error}")

    await session.start(agent=agent, room=ctx.room)
    session.say("말씀하십시오. 시스템이 준비되었습니다.", allow_interruptions=True)
