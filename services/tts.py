import os
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    Timeout,
)
import os
import uuid
import asyncio
import logging
import uuid
import httpx
from pathlib import Path
from typing import Optional
import soundfile as sf

# 加载 .env 文件中的环境变量
load_dotenv()
TTS_BASE_URL = os.getenv("TTS_BASE_URL")
TTS_MODEL_NAME = os.getenv("TTS_MODEL")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

tts_client = AsyncOpenAI(api_key="EMPTY", base_url=TTS_BASE_URL)


async def ai_to_voice(ai_reply: str) -> str:
    # 创建输出目录
    output_dir = "static/audios"
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, filename)

    # 使用新版 OpenAI SDK 流式接口
    async with tts_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL_NAME,
        voice="vivian",
        input=ai_reply,
    ) as response:

        await response.stream_to_file(output_path)

        if not os.path.exists(output_path):
            logger.error(f"❌ TTS文件不存在: {output_path}")
            raise RuntimeError(f"TTS文件不存在: {output_path}")

    return output_path


async def ai_to_voice_with_retry(
    ai_reply: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_retry_delay: float = 10.0,
) -> Optional[str]:
    """
    带重试机制的文本转语音

    Args:
        ai_reply: LLM生成的文本
        max_retries: 最大重试次数
        retry_delay: 初始重试间隔（秒）
        max_retry_delay: 最大重试间隔（秒）

    Returns:
        wav文件路径，失败返回None
    """
    # ============ 参数验证 ============
    if not ai_reply or not isinstance(ai_reply, str):
        logger.error(f"❌ 无效的AI回复文本: {type(ai_reply)}")
        return None

    # 清理文本（去除多余空白）
    ai_reply = ai_reply.strip()
    if not ai_reply:
        logger.error("❌ AI回复文本为空")
        return None

    # 限制文本长度
    max_length = 5000
    if len(ai_reply) > max_length:
        logger.warning(f"⚠️ 文本过长 ({len(ai_reply)}字符)，截断到{max_length}字符")
        ai_reply = ai_reply[:max_length]

    # ============ 准备输出目录 ============
    try:
        output_dir = "static/audios"
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"❌ 创建输出目录失败: {e}")
        return None

    # ============ 重试循环 ============
    retry_count = 0
    last_error = None
    output_path = None

    while retry_count < max_retries:
        # 每次重试生成新的文件名
        filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(output_dir, filename)

        try:
            logger.info(f"🔊 TTS尝试 (第{retry_count + 1}/{max_retries}次): {filename}")

            async with tts_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL_NAME, voice="vivian", input=ai_reply, timeout=30.0
            ) as response:

                # 检查响应
                if response.status_code != 200:
                    error_text = (
                        await response.text()
                        if hasattr(response, "text")
                        else "未知错误"
                    )
                    logger.error(
                        f"❌ TTS API错误: 状态码 {response.status_code}, {error_text}"
                    )
                    raise APIError(
                        message=f"❌ TTS API返回错误: {response.status_code} - {error_text}",
                        request=response.request,  # 从 response 中获取 request
                        body={"status_code": response.status_code, "error": error_text},
                    )

                # 流式写入文件
                await response.stream_to_file(output_path)

                # 验证文件
                if not await _validate_audio_file(output_path):
                    raise RuntimeError("生成的音频文件无效")

                logger.info(f"✅ TTS转换成功（第{retry_count + 1}次）: {output_path}")
                return output_path

        # ============ 可重试的错误 ============
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            wait_time = min(retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.warning(
                f"⚠️ 速率限制 (第{retry_count}次): {e}, "
                f"等待 {wait_time:.2f}秒后重试"
            )
            _cleanup_file(output_path)
            if retry_count < max_retries:
                await asyncio.sleep(wait_time)

        except (APIConnectionError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            retry_count += 1
            last_error = e
            wait_time = min(retry_delay * retry_count, max_retry_delay)
            logger.warning(
                f"⚠️ 连接/超时错误 (第{retry_count}次): {e}, "
                f"等待 {wait_time:.2f}秒后重试"
            )
            _cleanup_file(output_path)
            if retry_count < max_retries:
                await asyncio.sleep(wait_time)

        # ============ 不可重试的错误 ============
        except APIError as e:
            logger.error(f"❌ API错误 (不可重试): {e}")
            _cleanup_file(output_path)
            return None

        except PermissionError as e:
            logger.error(f"❌ 权限错误 (不可重试): {e}")
            _cleanup_file(output_path)
            return None

        except OSError as e:
            logger.error(f"❌ 文件系统错误 (不可重试): {e}")
            _cleanup_file(output_path)
            return None

        except Exception as e:
            logger.error(f"❌ 未知错误 (第{retry_count + 1}次): {e}", exc_info=True)
            _cleanup_file(output_path)
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(retry_delay * retry_count, max_retry_delay)
                logger.info(f"⏰ 等待 {wait_time:.2f}秒后重试")
                await asyncio.sleep(wait_time)
            else:
                return None

    # 所有重试都失败
    logger.error(f"❌ TTS转换失败，已重试{max_retries}次，最后错误: {last_error}")
    return None


def _cleanup_file(file_path: Optional[str]):
    """清理文件"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"✅ 已清理文件: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ 清理文件失败: {e}")


async def _validate_audio_file(file_path: str) -> bool:
    """验证音频文件是否有效"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"❌ 文件不存在: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"❌ 文件为空: {file_path}")
            return False

        # 检查文件头（WAV文件以RIFF开头）
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"RIFF":
                logger.warning(f"❌ 文件可能不是有效的WAV格式: {header}")
                # 不直接返回False，因为有些TTS服务可能返回其他格式
                # 但wav文件通常以RIFF开头

        # 检查文件大小是否合理（至少100字节）
        if file_size < 100:
            logger.error(f"❌ 文件太小: {file_size}字节")
            return False

        # logger.info(f"✅ 文件验证通过: {file_path} ({file_size/1024:.2f}KB)")
        return True

    except Exception as e:
        logger.error(f"❌ 验证音频文件失败: {e}")
        return False
