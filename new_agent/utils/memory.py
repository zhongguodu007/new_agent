from langchain.memory import (
    ConversationBufferMemory, #存储所有的对话历史
    ConversationBufferWindowMemory, #存储最近K轮的对话历史
    ConversationSummaryBufferMemory, #结合了缓冲区和摘要功能 适合长对话
    ConversationSummaryMemory, #存储所有的对话摘要，需要使用大模型生成摘要
)
from langchain.memory.summary import SummarizerMixin
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage,BaseMessage,get_buffer_string
from typing import List
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.utils import pre_init
from typing import Any, Dict, Optional, Tuple
import json
import aiofiles
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import BasePromptTemplate

class ContextMessage(HumanMessage):
    type: str = "context"

DEFAULT_SUMMARIZER_TEMPLATE = """逐步总结提供的对话内容，在之前摘要的基础上进行补充，生成新的摘要。

示例
当前摘要：
人类询问人工智能对人工智能的看法。人工智能认为人工智能是积极的力量。

新的对话内容：
人类：为什么你认为人工智能是积极的力量？  
人工智能：因为人工智能将帮助人类充分发挥潜力。

新的摘要：
人类询问人工智能对人工智能的看法。人工智能认为人工智能是积极的力量，因为它将帮助人类充分发挥潜力。
示例结束

当前摘要：
{summary}

新的对话内容：
{new_lines}

新的摘要：
"""
SUMMARY_PROMPT1 = PromptTemplate(
    input_variables=["summary", "new_lines"], template=DEFAULT_SUMMARIZER_TEMPLATE
)

class MemoryChain(BaseChatMemory, SummarizerMixin):
    
    # self.chat_memory.message :List[BaseMessage] #存储所有的对话历史
    summary_message: BaseMessage = SystemMessage(content="") #存储所有的对话摘要
    custom_context: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory
    ) #存储自定义上下文
    k: int = 5# 滑动窗口大小
    max_token_limit: int = 1500 # 摘要的最大长度
    single_limit:int = 1000 # 每个自定义上下文的最大长度
    memory_key: str = "History"#
    moving_summary_buffer: str = ""
    prompt: BasePromptTemplate = SUMMARY_PROMPT1

    @property
    def memory_cofig(self) -> Dict[str, Any]:
        return {
            "human_prefix": self.human_prefix,
            "ai_prefix": self.ai_prefix,
            "memory_key": self.memory_key,
            "k": self.k,
            "max_token_limit": self.max_token_limit,
            "single_limit": self.single_limit
        }
    def _chat_memory_as_str(self, messages: List[BaseMessage]) -> str:

        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
    
    def _all_messages_as_str(self)->str:

        string_messages = []
        for messsage in self.custom_context.messages:
            m = f"{messsage.type}: {messsage.content}"
            string_messages.append(m)

        all_context = "\n".join(string_messages)

        return "\n".join([f"History: {self.summary_message.content}", self._chat_memory_as_str(self.chat_memory.messages), all_context ])
    

    def _all_messages_as_dict(self)->List[Dict[str, str]]:
        string_messages = []
        string_messages.append({self.memory_key:self.summary_message.content})

        for messsage in self.chat_memory.messages:
            if isinstance(messsage, AIMessage):
                m = {self.ai_prefix: messsage.content}
            elif isinstance(messsage, HumanMessage):
                m = {self.human_prefix: messsage.content}
            else:
                raise ValueError(f"Unexpected message type: {type(messsage)}")
            string_messages.append(m)
        for messsage in self.custom_context.messages:
            string_messages.append({messsage.type:messsage.content})
            
        return string_messages

    def load_memory(self, inputs:List[Dict[str, str]]):

        for message in inputs:
            if self.ai_prefix in message.keys():
                self.chat_memory.add_messages([AIMessage(content=message[self.ai_prefix])])
            elif self.human_prefix in message.keys():
                self.chat_memory.add_messages([HumanMessage(content=message[self.human_prefix])])
            elif self.memory_key in message.keys():
                self.summary_message = self.summary_message_cls(content=message[self.memory_key])
            else:
                for key, value in message.items():
                    self.custom_context.add_messages([ContextMessage(type=key,content=value)])

    async def aload_memory(self, inputs: List[Dict[str, str]]):
        for message in inputs:
            if self.ai_prefix in message:
                # 添加 AI 消息到 chat_memory
                await self.chat_memory.aadd_messages([AIMessage(content=message[self.ai_prefix])])
            elif self.human_prefix in message:
                # 添加用户消息到 chat_memory
                await self.chat_memory.aadd_messages([HumanMessage(content=message[self.human_prefix])])
            elif self.memory_key in message:
                # 更新摘要消息
                self.summary_message = self.summary_message_cls(content=message[self.memory_key])
            else:
                # 处理自定义上下文消息
                for key, value in message.items():
                    await self.custom_context.aadd_messages([ContextMessage(type=key, content=value)])

    def _get_input_output(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        return list(inputs.values())[0], list(outputs.values())[0]
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)

        self.chat_memory.add_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )
        if len(self.chat_memory.messages) > self.k and self.llm.get_num_tokens_from_messages(self.chat_memory.messages) > self.max_token_limit:
            prune = []
            while self.llm.get_num_tokens_from_messages(self.chat_memory.messages) > self.max_token_limit:
                prune.append(self.chat_memory.messages.pop(-1))
                prune.append(self.chat_memory.messages.pop(-1))
            self.moving_summary_buffer = self.predict_new_summary(prune, self.moving_summary_buffer)
            self.summary_message = self.summary_message_cls(content=self.moving_summary_buffer)

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        await self.chat_memory.aadd_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )
        if len(self.chat_memory.messages) > self.k and self.llm.get_num_tokens_from_messages(self.chat_memory.messages) > self.max_token_limit:
            prune = []
            while self.llm.get_num_tokens_from_messages(self.chat_memory.messages) > self.max_token_limit:
                prune.append(self.chat_memory.messages.pop(0))
                prune.append(self.chat_memory.messages.pop(0))
            self.moving_summary_buffer = await self.apredict_new_summary(prune, self.moving_summary_buffer)
            self.summary_message = self.summary_message_cls(content=self.moving_summary_buffer)

    def save_custom_context(self, outputs: str, types:str) -> None:
        """Save context from this conversation to buffer."""
        length = self.llm.get_num_tokens(outputs)
        if length > self.single_limit:
            message = ContextMessage(type=types, content=outputs)
            context = self.predict_new_summary(messages=[message])
        else:
            context = outputs
        
        self.custom_context.add_messages(
            [ContextMessage(type=types, content=context)]
        )

    async def asave_custom_context(self, outputs: Dict[str, str], types:str) -> None:
        """Save context from this conversation to buffer."""
        length = self.llm.get_num_tokens_from_messages([outputs])
        if length > self.single_limit:
            buffer = await self.apredict_new_summary(messages=[outputs])
        else:
            buffer = outputs.values()
        
        await self.custom_context.aadd_messages(
            [ContextMessage(type=types, content=buffer)]
        )

    def delete_context(self, types:str) -> None:
        """Delete context from this conversation to buffer."""
        filter_context = [message for message in self.custom_context.messages if message.type != types]
        self.custom_context.clear()
        self.custom_context.add_messages(filter_context)
    
    async def delete_context(self, types: str) -> None:
        """异步删除指定类型的消息（假设方法是同步的）"""
        filter_context = [
            msg for msg in self.custom_context.messages if msg.type != types
        ]
        
        # 同步执行 clear 和 add_messages
        await self.custom_context.aclear()
        await self.custom_context.aadd_messages(filter_context)
       
    def get_contxt_types(self) -> List[str]:
        """Return all context types."""
        return [message.type for message in self.custom_context.messages]
    
    async def aget_contxt_types(self) -> List[str]:
        """Return all context types."""
        messages = await self.custom_context.aget_messages()
        return [message.type for message in messages]

    def get_all_memory(self) -> List[BaseMessage]:
        """Return history buffer."""
        return [self.summary_message] + self.chat_memory.messages + self.custom_context.messages
    
    async def aget_all_memory(self) -> List[BaseMessage]:
        """Return history buffer."""
        chat_messages = await self.chat_memory.aget_messages()
        summary_message = [self.summary_message]
        custom_context = await self.custom_context.aget_messages()
        return summary_message + chat_messages + custom_context
    
    async def aget_summary_message(self) -> BaseMessage:
        """Return history buffer."""
        return self.summary_message
    
    def get_summary_message(self) -> BaseMessage:
        """Return history buffer."""
        return self.summary_message

    @pre_init
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values
    
    def clear(self) -> None:
        super().clear()
        self.custom_context.clear()
        self.summary_message = []
        self.moving_summary_buffer = ""

    async def aclear(self) -> None:
        await super().aclear()
        await self.custom_context.aclear()
        self.summary_message = []
        self.moving_summary_buffer = ""
    
    def save_memory_to_file(self, file_path: str) -> None:
        """Save memory to a file."""

        memory_cofig = self.memory_cofig
        memory = self._all_messages_as_dict() # List[Dict[str, str]]
        all_info = [memory_cofig] + memory
        json.dump(all_info, open(file_path, "w", encoding="utf-8"), ensure_ascii=False)

    async def asave_memory_to_file(self, file_path: str) -> None:
        memory_cofig = self.memory_cofig
        memory =self._all_messages_as_dict()
        all_info = [memory_cofig] + memory
        json_content = json.dumps(all_info, ensure_ascii=False, indent=2)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json_content)

    def load_memory_from_file(self, file_path: str) -> None:
        """Load memory from a file."""
        all_info = json.load(open(file_path, "r", encoding="utf-8"))
        self.human_prefix = all_info[0]["human_prefix"]
        self.ai_prefix = all_info[0]["ai_prefix"]
        self.memory_key = all_info[0]["memory_key"]
        self.k = all_info[0]["k"]
        self.max_token_limit = all_info[0]["max_token_limit"]
        self.single_limit = all_info[0]["single_limit"]
        self.load_memory(all_info[1:])

        
    def load_memory_variables(self) -> dict:
        """实现具体逻辑，例如从内存中加载变量"""
        # 示例：返回一个包含内存变量的字典
        return {"memory_key": "memory_value"}
    
    def memory_variables(self) -> List[str]:
        """返回所有的内存变量"""
        return ["memory_key"]

from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
if __name__ == "__main__":
    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=1 
    )
    memory = MemoryChain(llm=llm)
    memory.load_memory([{"AI":"你好"},{"Human":"你好"},{"History":"昨天是2025年4月17日"}])
    memory.save_context({"input":"101"}, {"output":"102"})
    memory.save_custom_context("今天是2025年4月18日", types="date")
    print(memory.get_all_memory())
    memory.save_memory_to_file("./chat/测试对话.json")
    memory.clear()
    print('After clear: ',memory.get_all_memory())
    memory.load_memory_from_file("./chat/测试对话.json")
    print('After load: ',memory.get_all_memory())
    