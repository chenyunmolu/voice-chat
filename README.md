# 🎤 Voice LLM Assistant

基于 **FastAPI + FunASR + Qwen3.5 + Qwen3-TTS + vLLM** 构建的端到端实时语音助手。

项目实现了完整的 **语音输入 → 语音识别（ASR）→ 大语言模型（LLM）→ Tool Calling → 语音合成（TTS）** 流程，并预留了 RAG、数据库查询、Agent 等扩展能力。

---

## ✨ Features

- 🎤 Speech-to-Text（ASR）
  - 基于 FunASR
  - 支持 wav、mp3、m4a、aac、ogg、opus 等音频格式
  - 支持自动重试

- 🤖 Large Language Model
  - 基于 Qwen3.5
  - OpenAI API Compatible
  - 支持 Function Calling
  - 支持关闭 Thinking Mode

- 🔧 Tool Calling
  - 自动天气查询
  - 数据库查询（可扩展）
  - RAG 检索（可扩展）
  - Web Search（可扩展）

- 🔊 Text-to-Speech
  - 基于 Qwen3-TTS
  - 自动生成 wav
  - 返回可访问 URL

- 🚀 Backend
  - FastAPI
  - Async/Await
  - aiofiles
  - 自动重试机制
  - 静态资源托管

---

# Project Architecture

```
                 Microphone
                     │
                     ▼
              FunASR (ASR)
                     │
               Speech Text
                     │
                     ▼
             Qwen3.5-4B (LLM)
                     │
        ┌────────────┴────────────┐
        │                         │
   Direct Answer            Tool Calling
        │                         │
        │                 Weather / DB / RAG
        │                         │
        └────────────┬────────────┘
                     ▼
              Final Response
                     │
                     ▼
             Qwen3-TTS (TTS)
                     │
                     ▼
                 WAV Audio
```

---

# Project Structure

```
voice-chat
│
├── services
│   ├── asr.py              # ASR服务
│   ├── llm.py              # LLM服务
│   ├── tts.py              # TTS服务
│   ├── tools_call.py       # Tool Calling
│   └── __init__.py
│
├── static
│   ├── audios              # TTS生成音频
│   ├── images
│   └── index.html          # Demo页面
│
├── uploads                 # 临时上传目录
│
├── config.py               # 配置文件
├── main.py                 # FastAPI入口
├── .env
└── README.md
```

---

# Technology Stack

| Module | Technology |
|----------|------------|
| Backend | FastAPI |
| ASR | FunASR |
| LLM | Qwen3.5 |
| TTS | Qwen3-TTS |
| Inference | vLLM |
| HTTP Client | OpenAI SDK |
| Async | asyncio |
| File IO | aiofiles |

---

# Requirements

Python >= 3.11

推荐使用 uv 创建环境：

```bash
uv venv
source .venv/bin/activate
```

安装依赖：

```bash
uv pip install -r requirements.txt
```

---

# Model Deployment

本项目默认使用三个独立的推理服务。

## ASR

```
http://127.0.0.1:8001
```

例如：

```
Fun-ASR-Nano-2512
```

启动示例：

```bash
vllm serve /root/autodl-tmp/Fun-ASR-Nano-2512-vllm \
    --gpu-memory-utilization 0.2 \
    --trust-remote-code \
    --port 8001
```
---

## LLM

```
http://127.0.0.1:8002
```

例如：

```
Qwen3.5-4B
```

启动示例：

```bash
vllm serve /root/autodl-tmp/Qwen3.5-4B \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --port 8002
```

---

## TTS

```
http://127.0.0.1:8003
```

例如：

```
Qwen3-TTS-12Hz-0.6B-CustomVoice
```

启动示例：

```bash
vllm serve /root/autodl-tmp/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --omni \ 
    --gpu-memory-utilization 0.2 \
    --trust-remote-code \
    --enforce-eager \
    --port 8003
```

---

# Configuration

`.env`

```text
SERVER_BASE_URL=http://127.0.0.1:8081/
```

用于返回生成音频的访问地址。

---

# Run

开发模式：

```bash
uvicorn main:app --host 0.0.0.0 --port 8081 --reload
```

生产模式：

```bash
gunicorn main:app \
    -w 5 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8081 \
    --timeout 120
```

---

# API

## Health Check

```
GET /ping
```

返回：

```json
{
    "message":"pong"
}
```

---

## Voice Chat

```
POST /tts
```

参数：

```
multipart/form-data

file=audio.wav
```

返回：

```json
{
    "filename":"demo.wav",
    "transcription":"你好",
    "ai_reply":"你好，请问有什么可以帮助你？",
    "wav_url":"http://127.0.0.1:8081/audios/xxxx.wav"
}
```

---

## Speech Recognition

```
POST /transcribe/
```

返回：

```json
{
    "filename":"test.wav",
    "transcription":"你好",
    "ai_reply":"你好，请问有什么帮助？"
}
```

---

# Tool Calling

当前支持：

- Weather Query

未来可扩展：

- Database Query
- RAG Retrieval
- Web Search
- Calendar
- Email
- File System
- MCP Tools

调用流程：

```
User
 │
 ▼
Qwen3.5
 │
 ▼
Tool Calling
 │
 ▼
Python Tool
 │
 ▼
Qwen3.5
 │
 ▼
Final Answer
```

---

# Retry Mechanism

项目内置统一重试机制：

支持：

- APIConnectionError
- Timeout
- RateLimitError

采用：

- Exponential Backoff

提高系统稳定性。

---

# Frontend

访问：

```
http://localhost:8081/
```

即可看到 Demo 页面。

页面展示：

- Pipeline
- Feature
- Streaming Demo

后续可扩展：

- WebRTC
- 实时录音
- Streaming Chat
- Streaming TTS

---

# Future Roadmap

计划支持：

- [ ] Streaming ASR
- [ ] Streaming Chat
- [ ] Streaming TTS
- [ ] LangGraph Agent
- [ ] MCP
- [ ] Memory
- [ ] RAG
- [ ] Database Agent
- [ ] Multi Tool Calling
- [ ] Voice Activity Detection (VAD)
- [ ] WebRTC
- [ ] Docker Deployment

---

# License

MIT License