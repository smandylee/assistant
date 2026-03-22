const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("faustDesktop", {
  startAgent: () => ipcRenderer.invoke("agent:start"),
  stopAgent: () => ipcRenderer.invoke("agent:stop"),
  sendChat: (payload) => ipcRenderer.invoke("chat:send", payload),
  speakText: (text) => ipcRenderer.invoke("tts:speak", text),
  onAgentLog: (handler) => {
    const wrapped = (_event, message) => handler(message);
    ipcRenderer.on("agent:log", wrapped);
    return () => ipcRenderer.removeListener("agent:log", wrapped);
  },
  onAgentStatus: (handler) => {
    const wrapped = (_event, status) => handler(status);
    ipcRenderer.on("agent:status", wrapped);
    return () => ipcRenderer.removeListener("agent:status", wrapped);
  },
  onProactiveChat: (handler) => {
    const wrapped = (_event, payload) => handler(payload);
    ipcRenderer.on("chat:proactive", wrapped);
    return () => ipcRenderer.removeListener("chat:proactive", wrapped);
  },
});
