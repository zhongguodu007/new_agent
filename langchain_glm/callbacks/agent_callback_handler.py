# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_glm.utils import History
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import DuckDuckGoSearchRun


class BaseToolOutput:
    """
    LLM 要求 Tool 的输出为 str，但 Tool 用在别处时希望它正常返回结构化数据。
    只需要将 Tool 返回值用该类封装，能同时满足两者的需要。
    基类简单的将返回值字符串化，或指定 format="json" 将其转为 json。
    用户也可以继承该类定义自己的转换方法。
    """

    def __init__(
        self,
        data: Any,
        format: str = "",
        data_alias: str = "",
        **extras: Any,
    ) -> None:
        self.data = data
        self.format = format
        self.extras = extras
        if data_alias:
            setattr(self, data_alias, property(lambda obj: obj.data))

    def __str__(self) -> str:
        if self.format == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        else:
            return str(self.data)

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
        self.intermediate_steps: List[Tuple[AgentAction, BaseToolOutput]] = []
        self.outputs: Dict[str, Any] = {}

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        data = {
            "status": AgentStatus.llm_start,
            "text": "",
        }
        self.out = False
        self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        special_tokens = ["\nAction:", "\nObservation:", "<|observation|>"]
        for stoken in special_tokens:
            if stoken in token:
                before_action = token.split(stoken)[0]
                data = {
                    "status": AgentStatus.llm_new_token,
                    "text": before_action + "\n",
                }
                self.queue.put_nowait(dumps(data))
                self.out = False
                break

        if token is not None and token != "" and not self.out:
            data = {
                "run_id": str(kwargs["run_id"]),
                "status": AgentStatus.llm_new_token,
                "text": token,
            }
            self.queue.put_nowait(dumps(data))

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.llm_start,
            "text": "",
        }
        self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        data = {
            "run_id": str(kwargs["run_id"]),
            "status": AgentStatus.llm_end,
            "text": response.generations[0][0].message.content,
        }

        self.queue.put_nowait(dumps(data))

    async def on_llm_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        data = {
            "status": AgentStatus.error,
            "text": str(error),
        }
        self.queue.put_nowait(dumps(data))

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.tool_start,
            "tool": serialized["name"],
            "tool_input": input_str,
        }
        self.done.clear()
        self.queue.put_nowait(dumps(data))

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.tool_end,
            "tool": kwargs["name"],
            "tool_output": str(output),
        }
        self.queue.put_nowait(dumps(data))

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.error,
            "tool_output": str(error),
            "is_error": True,
        }

        self.queue.put_nowait(dumps(data))

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.agent_action,
            "action": {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            },
        }
        self.queue.put_nowait(dumps(data))

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if "Thought:" in finish.return_values["output"]:
            finish.return_values["output"] = finish.return_values["output"].replace(
                "Thought:", ""
            )

        data = {
            "run_id": str(run_id),
            "status": AgentStatus.agent_finish,
            "finish": {
                "return_values": finish.return_values,
                "log": finish.log,
            },
        }

        self.queue.put_nowait(dumps(data))

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        if "agent_scratchpad" in inputs:
            del inputs["agent_scratchpad"]
        if "chat_history" in inputs:
            inputs["chat_history"] = [
                History.from_message(message).to_msg_tuple()
                for message in inputs["chat_history"]
            ]
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.chain_start,
            "inputs": inputs,
            "parent_run_id": parent_run_id if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

        self.done.clear()
        self.out = False
        self.queue.put_nowait(dumps(data))

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.error,
            "error": str(error),
        }
        self.queue.put_nowait(dumps(data))

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if "intermediate_steps" in outputs:
            self.intermediate_steps = outputs["intermediate_steps"]
            self.outputs = outputs
            del outputs["intermediate_steps"]
        data = {
            "run_id": str(run_id),
            "status": AgentStatus.chain_end,
            "outputs": outputs,
            "parent_run_id": parent_run_id,
            "tags": tags,
        }
        self.queue.put_nowait(dumps(data))
        self.out = True
        # self.done.set()


if __name__ == "__main__":
    tools = [DuckDuckGoSearchRun(name="DuckDuckGo Search")]
    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    agent = initialize_agent(
    tools,
    llm,  # 选择模型
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    )
    async def run_agent():
        handler = AgentExecutorAsyncIteratorCallbackHandler()
        
        # 执行代理任务（异步方式）
        await agent.arun(
            "查询 Python 教程",
            callbacks=[handler]  # 注册回调处理器
        )
        
        # 处理事件队列
        while True:
            try:
                data_str = await handler.queue.get()
                data = json.loads(data_str)
                print(f"Received event: {data}")
                # ... 同前处理逻辑 ...
                handler.queue.task_done()
            except asyncio.QueueEmpty:
                break

    asyncio.run(run_agent())