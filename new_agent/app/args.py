from langchain_community.chat_models import ChatZhipuAI
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from new_agent.utils.memory import MemoryChain
from local import Agent_DB

# 智谱大模型
zhipu_llm =  ChatZhipuAI(
            model_name="glm-4-0520",
            api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            temperature=1
        )

# 聊天记忆模块
chat_memory = MemoryChain(
            memory_key="chat_history",
            return_messages=True,
            llm=zhipu_llm
        )

# 向量数据库
vectorstore = Agent_DB()

# 思维链回调管理器
callback_mamager = AsyncCallbackManager(
            [AgentExecutorAsyncIteratorCallbackHandler()]
        )

