import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import uuid

from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.schema import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_glm.agent_toolkits import BaseToolOutput
from new_agent.utils.history import History

def dumps(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

class AgentStatus:
    chain_start: int = 0
    llm_start: int = 1
    llm_new_token: int = 2
    llm_end: int = 3
    agent_action: int = 4
    agent_finish: int = 5
    tool_start: int = 6
    tool_end: int = 7
    error: int = -1
    chain_end: int = -999

class AgentExecutorAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.out = False
        self.intermediate_step: List[Tuple[AgentAction, BaseToolOutput]] = []
        self.outputs: Dict[str, Any] = {}

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"[{run_id}] 推理链启动！输入：{parent_run_id}")
        # if "agent_scratchpad" in inputs:
        #     del inputs["agent_scratchpad"]
        if "chat_history" in inputs:
            inputs["chat_history"] = [
                History.from_message(message).to_msg_tuple()
                for message in inputs["chat_history"]
            ]
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.chain_start,
            "inputs": inputs,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

        self.done.clear()
        self.out = False
        self.queue.put_nowait(dumps(data))

    async def on_chain_end(self, outputs, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] 推理链结束！输出：{parent_run_id}")
    
    async def on_chain_error(self, error, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] 推理出现错误！错误：{parent_run_id}")

    async def on_agent_action(self, action, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] Agent行动！Agent：{parent_run_id}")

    async def on_agent_finish(self, finish, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] Agent完成！Agent：{parent_run_id}")

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"[{run_id}] 聊天模型生成结果：{parent_run_id}")

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id = None, tags = None, metadata = None, inputs = None, **kwargs):
        print(f"[{run_id}] 模型调用工具：{parent_run_id}")

    async def on_tool_end(self, output, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] 模型调用工具输出：{parent_run_id}")

    async def on_tool_error(self, error, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] 模型调用工具出现错误：{parent_run_id}")
    
    async def on_retriever_start(self, serialized, query, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"[{run_id}] RAG开始！问询：{parent_run_id}")
    
    async def on_retriever_end(self, documents, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] RAG结束！文档：{parent_run_id}")
    async def on_retriever_error(self, error, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(f"[{run_id}] RAG出现错误！文档：{parent_run_id}")


