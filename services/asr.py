import os
import asyncio
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from pathlib import Path

# 加载 .env 文件中的环境变量
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ASR_BASE_URL = os.getenv("ASR_BASE_URL")
ASR_MODEL_NAME = os.getenv("ASR_MODEL")

asr_client = AsyncOpenAI(api_key="EMPTY", base_url=ASR_BASE_URL)


async def audio_to_text(audio_path: str):
    try:
        with open(audio_path, "rb") as audio_file:
            response = await asr_client.audio.transcriptions.create(
                model=ASR_MODEL_NAME,
                file=audio_file,
            )
        return response.text
    except Exception as e:
        logger.error(f"❌ 转录音频时发生错误 [audio_to_text]: {e}")
        return None


async def audio_to_text_with_retry(
    audio_path: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_retry_delay: float = 10.0,
) -> Optional[str]:
    """
    带重试机制的音频转文本

    Args:
        audio_path: 音频文件路径
        max_retries: 最大重试次数
        retry_delay: 初始重试间隔（秒）
        max_retry_delay: 最大重试间隔（秒）

    Returns:
        转换后的文本，失败返回None
    """
    # ============ 参数验证 ============
    if not audio_path or not isinstance(audio_path, str):
        logger.error(f"❌ 无效的音频路径: {audio_path}")
        return None

    if not os.path.exists(audio_path):
        logger.error(f"❌ 音频文件不存在: {audio_path}")
        return None

    if not os.path.isfile(audio_path):
        logger.error(f"❌ 路径不是文件: {audio_path}")
        return None

    # 检查文件大小
    try:
        file_size = os.path.getsize(audio_path)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            logger.error(f"❌ 音频文件过大: {file_size / (1024*1024):.2f}MB")
            return None
        if file_size == 0:
            logger.error(f"❌ 音频文件为空: {audio_path}")
            return None
    except OSError as e:
        logger.error(f"❌ 无法获取文件信息: {e}")
        return None

    # 检查文件格式
    valid_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".webm",
        ".opus",
    }
    file_ext = Path(audio_path).suffix.lower()
    if file_ext not in valid_extensions:
        logger.error(f"❌ 不支持的音频格式: {file_ext}")
        return None

    # ============ 重试循环 ============
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            logger.info(
                f"🎵 尝试转录音频 (第{retry_count + 1}/{max_retries}次): {audio_path}"
            )

            # 读取并转录音频
            with open(audio_path, "rb") as audio_file:
                response = await asr_client.audio.transcriptions.create(
                    model=ASR_MODEL_NAME,
                    file=audio_file,
                    timeout=60.0,  # 设置超时
                    # 可选的额外参数
                    # response_format="text",
                    # language="zh",
                    # temperature=0.0
                )

            # 验证响应
            if not response:
                logger.error("❌ API返回空响应")
                raise ValueError("空响应")

            if not hasattr(response, "text") or not response.text:
                logger.error(f"❌ 转录返回空文本 (第{retry_count + 1}次)")
                # 空文本不重试，直接返回None
                return None

            # 清理文本（去除首尾空白）
            text = response.text.strip()
            if not text:
                logger.error(f"❌ 转录文本只有空白字符 (第{retry_count + 1}次)")
                return None

            logger.info(
                f"✅ 转录成功 (第{retry_count + 1}次)，识别文本内容：{text}，文本长度: {len(text)}字符"
            )
            return text

        # ============ 可重试的错误 ============
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            # 速率限制使用指数退避
            wait_time = min(retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.warning(
                f"⚠️ 速率限制 (第{retry_count}次): {e}, "
                f"等待 {wait_time:.2f}秒后重试"
            )
            if retry_count < max_retries:
                await asyncio.sleep(wait_time)

        except (APIConnectionError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            retry_count += 1
            last_error = e
            wait_time = min(retry_delay * retry_count, max_retry_delay)
            logger.warning(
                f"⚠️  连接/超时错误 (第{retry_count}次): {e}, "
                f"等待 {wait_time:.2f}秒后重试"
            )
            if retry_count < max_retries:
                await asyncio.sleep(wait_time)

        # ============ 不可重试的错误 ============
        except APIError as e:
            # API错误（如认证失败、参数错误等）
            logger.error(f"❌ API错误 (不可重试): {e}")
            return None

        except FileNotFoundError:
            logger.error(f"❌ 文件在重试过程中被删除: {audio_path}")
            return None

        except PermissionError:
            logger.error(f"❌ 权限错误 (不可重试): {audio_path}")
            return None

        except (IOError, OSError) as e:
            logger.error(f"❌  文件IO错误 (不可重试): {e}")
            return None

        except ValueError as e:
            # 参数错误等
            logger.error(f"❌ 值错误 (不可重试): {e}")
            return None

        except Exception as e:
            # 未知错误
            logger.error(f"❌ 未知错误 (第{retry_count + 1}次): {e}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(retry_delay * retry_count, max_retry_delay)
                logger.info(f"等待 {wait_time:.2f}秒后重试")
                await asyncio.sleep(wait_time)
            else:
                return None

    # 所有重试都失败
    logger.error(f"❌ 转录失败，已重试{max_retries}次，最后错误: {last_error}")
    return None
