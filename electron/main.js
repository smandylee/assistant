const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

let mainWindow = null;
let agentProcess = null;
let chatWorkerProcess = null;
let chatWorkerBuffer = "";
let chatWorkerSeq = 1;
const chatWorkerPending = new Map();

function loadEnvFile() {
  const envPath = path.join(__dirname, "..", ".env");
  if (!fs.existsSync(envPath)) return;
  const text = fs.readFileSync(envPath, "utf-8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx <= 0) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (!key) continue;
    process.env[key] = value;
  }
}

function resolvePythonPath() {
  const venvPython = path.join(__dirname, "..", ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) return venvPython;
  return "python";
}

function pythonEnv() {
  return {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  };
}

function ensureChatWorker() {
  if (chatWorkerProcess) return chatWorkerProcess;
  const python = resolvePythonPath();
  const workerScript = path.join(__dirname, "..", "faust_chat_worker.py");

  chatWorkerProcess = spawn(python, [workerScript], {
    cwd: path.join(__dirname, ".."),
    env: pythonEnv(),
    stdio: ["pipe", "pipe", "pipe"],
  });

  chatWorkerProcess.stdout.setEncoding("utf8");
  chatWorkerProcess.stdout.on("data", (chunk) => {
    chatWorkerBuffer += chunk;
    while (true) {
      const idx = chatWorkerBuffer.indexOf("\n");
      if (idx < 0) break;
      const line = chatWorkerBuffer.slice(0, idx).trim();
      chatWorkerBuffer = chatWorkerBuffer.slice(idx + 1);
      if (!line) continue;

      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }

      if (msg.type === "response" && msg.id) {
        const pending = chatWorkerPending.get(msg.id);
        if (pending) {
          chatWorkerPending.delete(msg.id);
          pending.resolve(msg);
        }
        continue;
      }

      if (msg.type === "proactive") {
        mainWindow?.webContents.send("chat:proactive", {
          message: msg.message || "",
          emotion: msg.emotion || null,
        });
        continue;
      }

      if (msg.type === "fatal") {
        mainWindow?.webContents.send("agent:log", `[chat-worker] fatal: ${msg.error}\n`);
      }
    }
  });

  chatWorkerProcess.stderr.setEncoding("utf8");
  chatWorkerProcess.stderr.on("data", (chunk) => {
    mainWindow?.webContents.send("agent:log", `[chat-worker] ${chunk}`);
  });

  chatWorkerProcess.on("close", () => {
    for (const pending of chatWorkerPending.values()) {
      pending.resolve({ ok: false, error: "chat_worker_closed" });
    }
    chatWorkerPending.clear();
    chatWorkerProcess = null;
    chatWorkerBuffer = "";
  });

  return chatWorkerProcess;
}

async function callChatWorker(message) {
  const proc = ensureChatWorker();
  const id = `m${chatWorkerSeq++}`;
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      chatWorkerPending.delete(id);
      resolve({ ok: false, error: "chat_timeout" });
    }, 60000);

    chatWorkerPending.set(id, {
      resolve: (msg) => {
        clearTimeout(timeout);
        if (!msg || msg.ok !== true) {
          resolve({ ok: false, error: msg?.error || "chat_failed" });
          return;
        }
        resolve({ ok: true, reply: msg.reply || "", emotion: msg.emotion || null });
      },
    });

    proc.stdin.write(JSON.stringify({ id, message }) + "\n");
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 980,
    height: 760,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
}

app.whenReady().then(() => {
  loadEnvFile();
  createWindow();
  ensureChatWorker();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (chatWorkerProcess) {
    chatWorkerProcess.kill();
    chatWorkerProcess = null;
  }
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("agent:start", async () => {
  if (agentProcess) return { ok: true, status: "already_running" };

  const python = resolvePythonPath();
  const scriptPath = path.join(__dirname, "..", "agent.py");
  agentProcess = spawn(python, [scriptPath, "dev"], {
    cwd: path.join(__dirname, ".."),
    env: pythonEnv(),
    stdio: ["ignore", "pipe", "pipe"],
  });

  agentProcess.stdout.on("data", (data) => {
    mainWindow?.webContents.send("agent:log", data.toString());
  });
  agentProcess.stderr.on("data", (data) => {
    mainWindow?.webContents.send("agent:log", data.toString());
  });
  agentProcess.on("close", (code) => {
    mainWindow?.webContents.send("agent:status", { running: false, code });
    agentProcess = null;
  });

  mainWindow?.webContents.send("agent:status", { running: true });
  return { ok: true, status: "started" };
});

ipcMain.handle("agent:stop", async () => {
  if (!agentProcess) return { ok: true, status: "not_running" };
  agentProcess.kill();
  agentProcess = null;
  mainWindow?.webContents.send("agent:status", { running: false });
  return { ok: true, status: "stopped" };
});

ipcMain.handle("chat:send", async (_event, message) => {
  const request = typeof message === "string" ? { message } : message || {};
  const text = String(request.message || "").trim();
  if (!text) return { ok: false, error: "empty_message" };
  return callChatWorker(text);
});

ipcMain.handle("tts:speak", async (_event, text) => {
  const payload = typeof text === "string" ? { text } : text || {};
  const value = String(payload.text || "").trim();
  if (!value) return { ok: false, error: "empty_text" };
  try {
    const audioBase64 = await synthesizeElevenLabs(value, payload.emotion || null);
    return { ok: true, audioBase64, mimeType: "audio/mpeg" };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

function buildVoiceSettingsFromEmotion(emotion) {
  const e = emotion || {};
  const label = String(e.label || "neutral");
  const intensity = Number(e.intensity || 0.35);

  let stability = Math.max(0.3, Math.min(0.85, 0.74 - intensity * 0.3));
  let similarity = Math.max(0.65, Math.min(0.95, 0.82 + intensity * 0.08));

  if (["angry", "frustrated", "annoyed"].includes(label)) stability = Math.max(0.3, stability - 0.05);
  else if (["calm", "analytical"].includes(label)) stability = Math.min(0.88, stability + 0.08);
  else if (["playful", "excited"].includes(label)) stability = Math.max(0.3, stability - 0.03);

  return {
    stability: Number(stability.toFixed(3)),
    similarity_boost: Number(similarity.toFixed(3)),
  };
}

async function synthesizeElevenLabs(text, emotion = null) {
  const apiKey = process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_API_KEY;
  if (!apiKey) {
    throw new Error("ELEVENLABS_API_KEY is not configured");
  }
  const voiceId = process.env.ELEVENLABS_VOICE_ID || "FKpLBDCIkrMlLHfQVK29";
  const modelId = process.env.ELEVENLABS_MODEL || "eleven_multilingual_v2";
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "xi-api-key": apiKey,
      "Content-Type": "application/json",
      Accept: "audio/mpeg",
    },
    body: JSON.stringify({
      text,
      model_id: modelId,
      voice_settings: buildVoiceSettingsFromEmotion(emotion),
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`ElevenLabs TTS failed (${resp.status}): ${errText}`);
  }
  const audioBuffer = Buffer.from(await resp.arrayBuffer());
  return audioBuffer.toString("base64");
}
