import os
import asyncio
import httpx
import logging
from openai import (
    OpenAI,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from .tools_call import TOOLS, get_weather
import json
import re
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

llm_client = AsyncOpenAI(api_key="EMPTY", base_url=LLM_BASE_URL)


async def text_to_ai(question: str):
    response = await llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个语音助手，请直接回答用户，不输出思考过程。",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=512,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return response.choices[0].message.content


async def text_to_ai_with_retry(
    question: str, max_retries: int = 3, retry_delay: float = 1.0
) -> Optional[str]:
    """
    带重试机制的AI调用函数

    Args:
        question: 用户问题
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）

    Returns:
        AI回复内容或None
    """
    if not question or not isinstance(question, str):
        logger.error("问题为空或者无效")
        return None

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            logger.info(f"调用AI接口，尝试第{retry_count + 1}次")
            response = await llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个语音助手，请直接回答用户，不输出思考过程。",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=512,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30.0,
            )

            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    logger.warning("⚠️ AI返回空内容")
                    return None

            logger.error("响应格式无效")
            return None

        except RateLimitError as e:
            retry_count += 1
            last_error = e
            wait_time = retry_delay * (2 ** (retry_count - 1))  # 指数退避
            logger.warning(f"速率限制，第{retry_count}次重试，等待{wait_time}秒")
            await asyncio.sleep(wait_time)

        except (APIConnectionError, httpx.TimeoutException) as e:
            retry_count += 1
            last_error = e
            logger.warning(f"连接错误，第{retry_count}次重试")
            await asyncio.sleep(retry_delay)

        except APIError as e:
            # API错误可能不可重试，直接返回
            logger.error(f"API错误: {e}")
            return None

        except Exception as e:
            logger.error(f"未知错误: {e}", exc_info=True)
            return None

    logger.error(f"重试{max_retries}次后仍然失败: {last_error}")
    return None


"""
添加了tool工具调用
"""


def parse_tool_call_from_content(content: str) -> Optional[Dict[str, Any]]:
    """
    解析 AI 返回的文本格式工具调用
    支持格式：
    1. <tool_call><function=get_weather><parameter=city>北京</parameter></function></tool_call>
    2. 自定义 JSON 格式
    """
    if not content:
        return None

    # 方法1：使用正则表达式解析 XML 标签
    tool_pattern = r"<tool_call>.*?<function=(\w+)>.*?<parameter=(\w+)>(.*?)</parameter>.*?</function>.*?</tool_call>"
    match = re.search(tool_pattern, content, re.DOTALL)

    if match:
        tool_name = match.group(1)
        param_name = match.group(2)
        param_value = match.group(3).strip()
        return {"name": tool_name, "arguments": {param_name: param_value}}

    # 方法2：尝试解析为 JSON
    try:
        # 尝试提取 JSON 内容
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "tool" in data or "function" in data:
                return data
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析错误: {e}")

    return None


async def text_to_ai_with_tools(
    question: str, max_retries: int = 3, retry_delay: float = 1.0
) -> Optional[str]:
    if not question or not isinstance(question, str):
        logger.error("❌ 问题为空或者无效")
        return None

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            logger.info(
                f"🧠 调用AI接口 (第{retry_count + 1}/{max_retries}次): {question}"
            )
            # ==========================
            # 第一次LLM调用
            # 判断是否需要工具
            # ==========================
            response = await llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": """
                            你是一个语音助手。
                            如果需要查询实时信息，
                            请调用提供的工具。
                            不输出思考过程。
                            """,
                    },
                    {"role": "user", "content": question},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=512,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30.0,
            )
            if (
                response is None
                or not hasattr(response, "choices")
                or not response.choices
            ):
                logger.error("❌ AI第一次调用返回为空或者响应格式异常")
                return None
            message = response.choices[0].message

            if message is None:
                logger.error("❌ AI第一次调用message为空")
                return None

            # ==================================
            # 情况1：
            # 模型直接回答
            # ==================================
            # logger.info(f"🤖 模型第一次回答: {message}")
            if not message.tool_calls:
                if message.content:
                    logger.info(f"💬 模型直接回答：{message.content}")
                    return message.content
                else:
                    logger.warning("⚠️  模型直接回答：AI返回空文本")
                    return None
            # ==================================
            # 情况2：
            # 模型请求调用工具
            # ==================================

            if not isinstance(message.tool_calls, list) or len(message.tool_calls) == 0:
                logger.error("❌ 模型请求调用工具：tool_calls格式异常")
                return None

            tool_call = message.tool_calls[0]

            if (
                tool_call is None
                or not hasattr(tool_call, "function")
                or tool_call.function is None
            ):
                logger.error("❌ 模型请求调用工具：工具调用结构异常")
                return None
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"❌ 模型请求调用工具：工具参数JSON解析失败:{e}")
                return None
            logger.info(
                f"🛠️  模型请求调用工具：调用工具: {tool_name}, 参数:{arguments}"
            )

            # 执行工具
            if tool_name == "get_weather":
                tool_result = await get_weather(arguments["city"])
            else:
                tool_result = {"error": "未知工具"}

            if tool_result is None:
                logger.error("❌ 模型请求调用工具：工具执行返回为空")
                tool_result = {"error": "工具执行失败"}

            # ==================================
            # 第二次LLM调用
            # 让模型组织答案
            # ==================================
            second_response = await llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个语音助手，请根据工具结果回答用户。请直接回答用户，不输出思考过程。",
                    },
                    {"role": "user", "content": question},
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    },
                ],
                tools=TOOLS,
                temperature=0.7,
                max_tokens=512,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30.0,
            )

            if (
                second_response is None
                or not hasattr(second_response, "choices")
                or not second_response.choices
            ):
                logger.error("❌ AI第二次调用返回为空或者响应格式异常")
                return None
            final_message = second_response.choices[0].message

            if final_message is None or not final_message.content:
                logger.error("❌ AI第二次调用返回空内容")
                return None
            logger.info(f"💬 模型最终回答: {final_message.content}")
            return final_message.content
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            wait_time = retry_delay * (2 ** (retry_count - 1))
            logger.warning(f"⚠️ 速率限制，第{retry_count}次重试，等待{wait_time}秒")
            await asyncio.sleep(wait_time)
        except (APIConnectionError, httpx.TimeoutException) as e:
            retry_count += 1
            last_error = e
            logger.warning(f"⚠️ 连接错误，第{retry_count}次重试")
            await asyncio.sleep(retry_delay)
        except APIError as e:
            # API错误可能不可重试，直接返回
            logger.error(f"❌ API错误: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 未知错误: {e}", exc_info=True)
            return None

    logger.error(f"❌ 重试{max_retries}次后仍然失败: {last_error}")
    return None
