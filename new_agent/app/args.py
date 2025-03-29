from langchain.callbacks.manager import AsyncCallbackManager
import sys
import os
sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
#from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain_community.chat_models import ChatZhipuAI
#from new_agent.agents.all_agents import init_rag_agent, init_search_agent
from new_agent.utils.memory import MemoryChain
from new_agent.app.local import Agent_DB

#占位符，能引用了就删
init_rag_agent = init_search_agent = AgentExecutorAsyncIteratorCallbackHandler = 0

# 共享的全局依赖
llm = ChatZhipuAI(
    model_name="glm-4-0520",
    api_key="your_api_key",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=1
)

file_folder = "cache/"
db = Agent_DB(file_folder)
global_memory = MemoryChain(
    memory_key="chat_history",
    return_messages=True,
    llm=llm
)

#全局状态
current_agent = None
current_mode = None
