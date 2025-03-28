# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from new_agent.tools.struct_type import (
    AdapterAllToolStructType,
)
from new_agent.tools.tool import (
    AllToolExecutor,
    BaseToolOutput,
    AdapterAllTool
)
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


class WebBrowserToolOutput(BaseToolOutput):
    platform_params: Dict[str, Any]

    def __init__(
        self,
        data: Any,
        platform_params: Dict[str, Any],
        **extras: Any,
    ) -> None:
        super().__init__(data, "", "", **extras)
        self.platform_params = platform_params


@dataclass
class WebBrowserAllToolExecutor(AllToolExecutor):
    """platform adapter tool for code interpreter tool"""

    name: str

    def run(
        self,
        tool: str,
        tool_input: str,
        log: str,
        outputs: List[Union[str, dict]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> WebBrowserToolOutput:
        if outputs is None or str(outputs).strip() == "":
            raise ValueError(f"Tool {self.name}  is server error")

        return WebBrowserToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )

    async def arun(
        self,
        tool: str,
        tool_input: str,
        log: str,
        outputs: List[Union[str, dict]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> WebBrowserToolOutput:
        """Use the tool asynchronously."""
        if outputs is None or str(outputs).strip() == "" or len(outputs) == 0:
            raise ValueError(f"Tool {self.name}  is server error")

        return WebBrowserToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )


class WebBrowserAdapterAllTool(AdapterAllTool[WebBrowserAllToolExecutor]):
    @classmethod
    def get_type(cls) -> str:
        return "WebBrowserAdapterAllTool"

    def _build_adapter_all_tool(
        self, platform_params: Dict[str, Any]
    ) -> WebBrowserAllToolExecutor:
        return WebBrowserAllToolExecutor(
            name=AdapterAllToolStructType.WEB_BROWSER, platform_params=platform_params
        )

class WebBrowserTool(BaseTool):
    name = "web_browser"
    description = "用于执行浏览器操作，支持访问URL、查找元素、执行JavaScript等"

    def _run(self, command: str) -> str:
        """同步执行"""
        # 初始化适配器和执行器（需根据实际框架调整）
        adapter = WebBrowserAdapterAllTool()
        executor = adapter._build_adapter_all_tool({
            "browser_type": "chrome",
            "headless": True
        })
        output = executor.run(
            tool=self.name,
            tool_input=command,
            log="Starting execution"
        )
        return f"Result: {output.data}\nLog: {output.log}"

    async def _arun(self, command: str) -> str:
        """异步执行"""
        adapter = WebBrowserAdapterAllTool()
        executor = adapter._build_adapter_all_tool({
            "browser_type": "chrome",
            "headless": True
        })
        output = await executor.arun(
            tool=self.name,
            tool_input=command,
            log="Starting execution"
        )
        return f"Result: {output.data}\nLog: {output.log}"