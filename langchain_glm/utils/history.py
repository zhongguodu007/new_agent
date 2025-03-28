# -*- coding: utf-8 -*-
import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

from langchain.prompts.chat import ChatMessagePromptTemplate
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from zhipuai.core import BaseModel
from pydantic import model_validator

logger = logging.getLogger()


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            # if message_dict["content"] == "":
            #     message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            # if message_dict["content"] == "":
            #     message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """

    role: str
    content: str

    def to_msg_tuple(self):
        '''
        将角色映射为 (role, content)元组
        '''
        return "ai" if self.role == "assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        '''
        将消息字典转换成 prompt模板
        '''
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw:  # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )
    
    @model_validator(mode="after")
    def validate_content(self):
        if self.content is None:
            self.content = ""

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        '''
        将输入的列表、字典、元组转换成History对象
        '''
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1] if h[1] is not None else "")
        elif isinstance(h, dict):
            h = cls(**h)

        return h

    @classmethod
    def from_message(cls, message: BaseMessage) -> "History":
        '''
        将消息转换成History
        '''
        return cls.from_data(_convert_message_to_dict(message=message))


if __name__ == "__main__":
    hum_msg = HumanMessage(content="你好")
    ai_msg = AIMessage(content="",additional_kwargs={"function_call":{"name":"search", "argmuent":{"query":"Python教程"}}})
    # print(_convert_message_to_dict(hum_msg))
    print(_convert_message_to_dict(ai_msg))
    his = History.from_message(ai_msg)
    
    print(his)