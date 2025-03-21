# -*- coding: utf-8 -*-
import logging
import logging.config
import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
import requests
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = "df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc"

logging_conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'colored': {
            'format': '%(message)s',  # 这里仅输出消息部分，避免其他格式干扰ANSI转义序列
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'colored',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        }
    }
}

def test_openai_demo_2_tools():
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = OpenAI(
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[{"role": "user", "content": "帮我查询天气"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        top_p=0.7,
        temperature=0.1,
        max_tokens=2000,
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

@tool
def get_current_weather(location: str, unit: str = "celsius"):
    """获取指定地点的当前天气信息"""
    # 假设这是一个调用天气API的函数
    api_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={110101}&key=f23a5629053c50488bbd27cdbab74c8e"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "无法获取天气数据"}

def test_openai_demo1_tool_use():
    llm = ChatOpenAI(
        model="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        streaming=True,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )

    tools = [multiply, exponentiate, add, get_current_weather]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    def call_tools(msg: AIMessage) -> Runnable:
        """Simple sequential tool calling helper."""
        tool_map = {tool.name: tool for tool in tools}
        tool_calls = msg.tool_calls.copy()
        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls

    chain = llm_with_tools | call_tools
    out = chain.invoke(
        "北京今天的天气怎样"
    )
    print(out)

if __name__ == '__main__':
    test_openai_demo1_tool_use()
