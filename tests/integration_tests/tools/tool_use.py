# -*- coding: utf-8 -*-
from operator import itemgetter
from typing import Dict, List, Union
import sys
sys.path.append('C:\\Users\\DELL\\Desktop\\rag\\new-agent')
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import tool

from langchain_glm import ChatZhipuAI
import os
os.environ['ZHIPUAI_API_KEY'] = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc'


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


def test_tool_use():
    llm = ChatZhipuAI(model="glm-4-alltools", streaming=True)

    tools = [multiply, exponentiate, add]
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
        "What's 23 times 7, and what's five times 18 and add a million plus a billion and cube thirty-seven"
    )
    print(out)

if __name__ == '__main__':
    test_tool_use()