import json
import asyncio
from typing import Final


async def get_weather(city: str):
    """
    查询天气
    实际项目这里调用天气API
    """
    try:
        # 模拟异步请求天气API
        await asyncio.sleep(1)
        # 示例数据
        return {
            "success": True,
            "data": {
                "city": city,
                "temperature": "25°C",
                "condition": "晴",
                "humidity": "60%",
            },
        }

    except Exception as e:
        return {"success": False, "error": "天气接口失败: " + str(e)}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "当用户询问实时天气、温度、湿度、天气预报等问题时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如济南、上海",
                }
            },
            "required": ["city"],
        },
    },
}


async def query_database(user_id: str):
    """
    查询数据库
    实际项目这里调用数据库API
    """
    try:
        # 模拟异步请求数据库API
        await asyncio.sleep(1)
        # 示例数据
        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "name": "张三",
                "age": 30,
                "sex": "男",
                "email": "zhangsan@163.com",
            },
        }
    except Exception as e:
        return {"success": False, "error": "数据库接口失败: " + str(e)}


DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "当用户查询用户资料、个人信息、年龄、邮箱等数据库内容时调用。",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
}


# ==========================
# Tool Router
# ==========================
TOOL_ROUTER = {
    "get_weather": get_weather,
    "query_database": query_database,
}

TOOLS: Final = (
    WEATHER_TOOL,
    DATABASE_TOOL,
)
