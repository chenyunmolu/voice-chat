import os
from contextlib import asynccontextmanager
from fastapi import (
    Depends,
    FastAPI,
    Request,
    HTTPException,
    Header,
    BackgroundTasks,
    UploadFile,
    File,
)
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Generator
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
import asyncio
import aiofiles
from services.asr import audio_to_text, audio_to_text_with_retry
from services.llm import text_to_ai, text_to_ai_with_retry, text_to_ai_with_tools
from services.tts import ai_to_voice, ai_to_voice_with_retry
import timeit
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv()
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8081/")

# 定义生命周期


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================
    # 🟢 启动时执行
    LOGGER.info("🚀 服务器正在启动...")

    yield  # 服务运行中...

    # ==========================
    # ⛔ 关闭时执行 (可选)
    # ==========================
    LOGGER.info("⛔ 服务器正在关闭...")


app = FastAPI(title="Voice Chat Linux API", version="1.0", lifespan=lifespan)

# 允许 WebUI 来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="static/images"), name="images")
app.mount("/audios", StaticFiles(directory="static/audios"), name="audios")

limiter = Limiter(key_func=get_remote_address)


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/ping")
async def ping():
    return {"message": "pong"}


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # 1. 保存上传的音频文件
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 2. 调用模型进行转录
        LOGGER.info(f"开始转录: {file.filename}")
        transcription_text = await audio_to_text(file_path)
        LOGGER.info(f"语音 -> 文本: {transcription_text}")
        ai_reply = await text_to_ai(transcription_text)
        LOGGER.info(f"文本 -> AI: {ai_reply}")
        return {
            "filename": file.filename,
            "transcription": transcription_text,
            "ai_reply": ai_reply,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # 3. 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/tts")
async def transcribe_speech(file: UploadFile = File(..., max_size=100_000_000)):
    start_time = timeit.default_timer()
    ext = os.path.splitext(file.filename)[1]
    ALLOWED_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".webm",
        ".opus",
    }
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        # 1. 保存上传的音频文件
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())
        # 2. 调用模型进行转录
        transcription_text = await audio_to_text_with_retry(file_path)
        if transcription_text:
            ai_reply = await text_to_ai_with_tools(transcription_text)
            if ai_reply:
                wav_path = await ai_to_voice_with_retry(ai_reply)
                if wav_path:
                    SERVER_WAV_URL = SERVER_BASE_URL + wav_path
                    LOGGER.info(f"AI -> 音频: {SERVER_WAV_URL}")
                    return {
                        "filename": file.filename,
                        "transcription": transcription_text,
                        "ai_reply": ai_reply,
                        "wav_url": SERVER_WAV_URL,
                    }
                else:
                    LOGGER.warning(f"TTS  failed")
                    return {
                        "filename": file.filename,
                        "transcription": transcription_text,
                        "ai_reply": ai_reply,
                        "wav_url": "",
                    }
            else:
                LOGGER.warning(f"AI LLM  failed")
                return {
                    "filename": file.filename,
                    "transcription": transcription_text,
                    "ai_reply": "",
                    "wav_url": "",
                }
        else:
            LOGGER.warning(f"ASR  failed")
            return {
                "filename": file.filename,
                "transcription": "",
                "ai_reply": "",
                "wav_url": "",
            }
    except Exception as e:
        LOGGER.error(f"未知错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
        end_time = timeit.default_timer()
        LOGGER.info(f"总处理时间: {end_time - start_time:.2f} 秒")


"""
uvicorn main:app --host 0.0.0.0 --port 8081 --reload
gunicorn main:app -w 5 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8081 --timeout 120
"""
