const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const chatStatus = document.getElementById("chatStatus");
const agentLog = document.getElementById("agentLog");
const agentStatus = document.getElementById("agentStatus");
const emotionBadge = document.getElementById("emotionBadge");
const startAgentBtn = document.getElementById("startAgentBtn");
const stopAgentBtn = document.getElementById("stopAgentBtn");
const toggleMicBtn = document.getElementById("toggleMicBtn");
const toggleSpeechBtn = document.getElementById("toggleSpeechBtn");
const voiceStatus = document.getElementById("voiceStatus");

let speechOutputEnabled = true;
let isListening = false;
let recognition = null;
const outputAudio = new Audio();

function appendChat(role, text, cls) {
  const line = document.createElement("div");
  line.className = `chat-item ${cls}`;
  line.textContent = `[${role}] ${text}`;
  chatLog.appendChild(line);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function applyEmotion(emotion) {
  if (!emotion || !emotion.label) return;
  const intensity = Number(emotion.intensity ?? 0).toFixed(2);
  emotionBadge.textContent = `emotion: ${emotion.label} (${intensity})`;
}

async function sendChat(rawText) {
  const text = (rawText ?? chatInput.value).trim();
  if (!text) return;
  appendChat("나", text, "msg-user");
  if (rawText == null) chatInput.value = "";
  chatStatus.textContent = "응답 생성 중...";
  sendBtn.disabled = true;
  try {
    const res = await window.faustDesktop.sendChat({ message: text });
    if (res?.ok) {
      const reply = res.reply || "(빈 응답)";
      applyEmotion(res.emotion);
      appendChat("파우스트", reply, "msg-bot");
      chatStatus.textContent = "";

      // Do TTS asynchronously so text response appears immediately.
      if (speechOutputEnabled) {
        chatStatus.textContent = "음성 생성 중...";
        window.faustDesktop
          .speakText({ text: reply, emotion: res.emotion || null })
          .then(async (ttsRes) => {
            if (!ttsRes?.ok) {
              appendChat("TTS", ttsRes?.error || "음성 생성 실패", "msg-err");
              return;
            }
            outputAudio.src = `data:${ttsRes.mimeType || "audio/mpeg"};base64,${ttsRes.audioBase64}`;
            await outputAudio.play().catch(() => {});
          })
          .finally(() => {
            chatStatus.textContent = "";
          });
      }
    } else {
      appendChat("오류", res?.error || "알 수 없는 오류", "msg-err");
      chatStatus.textContent = "실패";
    }
  } catch (err) {
    appendChat("오류", String(err), "msg-err");
    chatStatus.textContent = "실패";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

function setupVoiceRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    voiceStatus.textContent = "브라우저 음성인식 미지원";
    toggleMicBtn.disabled = true;
    return;
  }

  recognition = new SR();
  recognition.lang = "ko-KR";
  recognition.continuous = true;
  recognition.interimResults = false;

  recognition.onstart = () => {
    isListening = true;
    toggleMicBtn.textContent = "음성 입력 중지";
    toggleMicBtn.classList.add("active");
    voiceStatus.textContent = "마이크 수신 중...";
  };

  recognition.onend = () => {
    isListening = false;
    toggleMicBtn.textContent = "음성 입력 시작";
    toggleMicBtn.classList.remove("active");
    voiceStatus.textContent = "마이크 대기";
  };

  recognition.onerror = (ev) => {
    voiceStatus.textContent = `음성 입력 오류: ${ev.error}`;
  };

  recognition.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (!result.isFinal) continue;
      const transcript = result[0]?.transcript?.trim();
      if (!transcript) continue;
      sendChat(transcript);
    }
  };
}

toggleMicBtn.addEventListener("click", () => {
  if (!recognition) return;
  if (isListening) recognition.stop();
  else recognition.start();
});

toggleSpeechBtn.addEventListener("click", () => {
  speechOutputEnabled = !speechOutputEnabled;
  toggleSpeechBtn.textContent = speechOutputEnabled ? "음성 출력 ON" : "음성 출력 OFF";
  toggleSpeechBtn.classList.toggle("active", speechOutputEnabled);
});

startAgentBtn.addEventListener("click", async () => {
  const res = await window.faustDesktop.startAgent();
  const status = res.status || "started";
  agentStatus.textContent = status;
  agentStatus.classList.toggle("status-running", status === "started" || status === "already_running");
});

stopAgentBtn.addEventListener("click", async () => {
  const res = await window.faustDesktop.stopAgent();
  const status = res.status || "stopped";
  agentStatus.textContent = status;
  agentStatus.classList.remove("status-running");
});

window.faustDesktop.onAgentLog((msg) => {
  agentLog.textContent += msg;
  agentLog.scrollTop = agentLog.scrollHeight;
});

window.faustDesktop.onAgentStatus((status) => {
  agentStatus.textContent = status.running ? "running" : "stopped";
  agentStatus.classList.toggle("status-running", !!status.running);
});

window.faustDesktop.onProactiveChat((payload) => {
  const message = String(payload?.message || "").trim();
  if (!message) return;
  if (payload?.emotion) applyEmotion(payload.emotion);
  appendChat("파우스트", message, "msg-bot");

  if (speechOutputEnabled) {
    window.faustDesktop
      .speakText({ text: message, emotion: payload?.emotion || null })
      .then(async (ttsRes) => {
        if (!ttsRes?.ok) return;
        outputAudio.src = `data:${ttsRes.mimeType || "audio/mpeg"};base64,${ttsRes.audioBase64}`;
        await outputAudio.play().catch(() => {});
      })
      .catch(() => {});
  }
});

setupVoiceRecognition();
