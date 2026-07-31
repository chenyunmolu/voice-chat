import json
import asyncio


async def get_weather(city: str):
    """
    查询天气
    实际项目这里调用天气API
    """

    # 示例数据
    return {"city": city, "weather": "晴", "temperature": "28℃", "humidity": "50%"}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市天气信息",
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

DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "查询用户数据库信息",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}},
    },
}

TOOLS = [WEATHER_TOOL]
