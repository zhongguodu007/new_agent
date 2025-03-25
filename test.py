from langchain_chroma import Chroma
from new_agent.retrival.base import NewRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.callbacks.manager import AsyncCallbackManager
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor,create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import subprocess
from typing import Union
from new_agent.tools.web_browser_tool import WebBrowserTool
from new_agent.agents.all_agents import init_rag_agent, init_search_agent

class WindowsTerminalTool(Tool):
    def __init__(self):
        super().__init__(
            name="Windows终端",  # 必填
            func=self._run,     # 必填：指向具体执行方法
            description="执行Windows终端命令，支持命令：dir, ipconfig..."  # 必填
        )
    async def _run(self, command: str) -> Union[str, Exception]:
        # 安全检查：仅允许白名单内的命令
        # cmd = command.strip().split()[0]
        # if cmd not in ALLOWED_COMMANDS:
        #     return f"错误：禁止执行命令 '{cmd}'，仅允许以下命令：{', '.join(ALLOWED_COMMANDS.keys())}"
        
        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10  # 防止长时间运行
            )
            #if result.returncode == 0:
            return f"输出：\n{result.stdout}"
            # else:
            #     return f"错误：{result.stderr}"
        except subprocess.TimeoutExpired:
            return "错误：命令执行超时"
        except Exception as e:
            return f"异常：{str(e)}"

    def _arun(self, command: str):
        # 异步执行（此处简化为同步）
        return self._run(command)

async def main3():
    callback_handler = AgentExecutorAsyncIteratorCallbackHandler()
    callback_manager = AsyncCallbackManager([callback_handler])
    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=1 
    )
    agent = init_search_agent(callback_manager=callback_manager,llm=llm)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    def get_user_input():
        user_input = input("User: ")
        return {
            "input": user_input,
            "chat_history": memory.chat_memory.messages  # 传递历史记录
        }

    def save_response(response):
        memory.save_context(
            inputs={"input": response["input"]},
            outputs={"output": response["output"]}
        )
    while True:
        user_input = get_user_input()
        try:
            response = await agent.ainvoke(user_input)
            print(f"Agent: {response['output']}")
            save_response(response)
        except Exception as e:
            print(f"Error: {e}")
async def main1():
    # 创建回调管理器并注册你的回调处理器
    callback_handler = AgentExecutorAsyncIteratorCallbackHandler()
    callback_manager = AsyncCallbackManager([callback_handler])
    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=1 
    )
    chroma_store = Chroma(
        collection_name="rag2",
        persist_directory="./rag"
    )
    retriever = NewRetriever(
        store=chroma_store,
        search_kwargs={"n_results":8}
    )
    prompt_template = ChatPromptTemplate.from_template(
    "使用以下上下文回答最后的问题。如果你不知道答案，请直接说你不知道，不要尝试编造答案。\n\n上下文:\n{context}\n\n问题: {question}"
    )
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # 或者选择其他适合的类型，如"map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt":prompt_template}
    )
    retrieval_tool = Tool(
        name="本地知识库",
        func=qa_chain.run,
        description="用于从本地知识库中检索相关信息以回答问题。适用于封闭域问题（如技术文档、内部资料等）。"
    )

    search_tool = DuckDuckGoSearchRun(
        name="DuckDuckGo Search",
        description="用于从互联网上搜索公开信息。适用于开放域问题（如最新新闻、教程等）。"
    )
    search = GoogleSearchAPIWrapper(
        google_api_key="AIzaSyCcmRzm1bFVENGV4-HOSbcLeT4zRo5LEyE",
        google_cse_id="72de5f772041b46fd"
    )
    search_tool1 = Tool(
            name="Google Search",
            func=search.run,
            description="适用于需要实时信息的问题回答。",
        )

   
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = ChatPromptTemplate.from_template(
        """
        你正在以一个智能体的身份运行。你拥有以下工具：
        {tools}
        此外，以下是之前的对话历史：
        {chat_history}

        使用这些工具来回答用户的问题：“{input}”。你之前已经进行了以下思考和行动：
        {agent_scratchpad}

        请严格从以下工具中选择一个执行：
        工具列表：{tool_names}

        输出格式必须包含：
        Thought: [你的思考过程]
        Action: [必须是工具列表中的名称（如：本地知识库 或 DuckDuckGo Search）]
        Action Input: [传递给工具的参数]

        结合工具的输出结果总结你的答案，如果没有确切答案可以使用不同的搜索工具来查找更多信息。
        如果问题已解决，请直接输出：
        Final Answer: [最终答案]
        """
    )

    agent = create_react_agent(
        llm=llm,
        tools=[search_tool1, search_tool],
        prompt=prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[search_tool1, search_tool],
        handle_parsing_errors=True,  # 自动处理解析错误
        callback_manager=AsyncCallbackManager(
            [AgentExecutorAsyncIteratorCallbackHandler()]
        ),
        verbose=True
    )
    def get_user_input():
        user_input = input("User: ")
        return {
            "input": user_input,
            "chat_history": memory.chat_memory.messages  # 传递历史记录
        }

    def save_response(response):
        memory.save_context(
            inputs={"input": response["input"]},
            outputs={"output": response["output"]}
        )
    while True:
        user_input = get_user_input()
        try:
            response = await agent_executor.ainvoke(user_input)
            print(f"Agent: {response['output']}")
            save_response(response)
        except Exception as e:
            print(f"Error: {e}")

     # 初始化代理（Agent）
    # agent = initialize_agent(
    #     tools=[search_tool, retrieval_tool],
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=False,
    #     callback_manager=callback_manager,
    #     # retriever=retriever
    # )

    # # 执行代理任务
    # try:
    #     result = await agent.ainvoke("我想知道A Novel Contrastive Signal Generative Framework for Accurate Graph Learning这篇文章的主要内容，这是在知识库的文章，用中文回答我的问题")
    #     print("最终结果：", result)
    # except Exception as e:
    #     print("执行代理时出错：", e)

    #question = "我想知道OBJECT-ORIENTED RELATIONAL DISTILLATION FOR OBJECT DETECTION这篇文章的主要内容，这是在知识库的文章，用中文回答我的问题"
    # question = "我想知道OBJECT-ORIENTED RELATIONAL DISTILLATION FOR OBJECT DETECTION这篇文章的具体方法是什么，这是在知识库的文章，用中文回答我的问题"
    # result = await agent_executor.ainvoke({"input": question})
    # print("最终答案：", result.get("output"))

def main2():
    # 初始化记忆模块（完整对话历史）
    memory = ConversationBufferMemory()
    callback_handler = AgentExecutorAsyncIteratorCallbackHandler()
    callback_manager = AsyncCallbackManager([callback_handler])
    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        temperature=1 
    )
    chroma_store = Chroma(
        collection_name="rag2",
        persist_directory="./rag"
    )
    retriever = NewRetriever(
        store=chroma_store,
        search_kwargs={"n_results":8}
    )
    prompt_template = ChatPromptTemplate.from_template(
    "使用以下上下文回答最后的问题。如果你不知道答案，请直接说你不知道，不要尝试编造答案。\n\n上下文:\n{context}\n\n问题: {question}"
    )
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # 或者选择其他适合的类型，如"map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt":prompt_template}
    )
    retrieval_tool = Tool(
        name="本地知识库",
        func=qa_chain.run,
        description="用于从本地知识库中检索相关信息以回答问题。适用于封闭域问题（如技术文档、内部资料等）。"
    )

    search_tool = DuckDuckGoSearchRun(
        name="DuckDuckGo Search",
        description="用于从互联网上搜索公开信息。适用于开放域问题（如最新新闻、教程等）。"
    )

    # 初始化Agent
    agent = initialize_agent(
        tools=[retrieval_tool, search_tool],
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,  # 添加记忆模块
        verbose=True,
        callback_manager=callback_manager,
        
    )

    # 执行对话
    agent.invoke({"input": "你好，今天过得怎么样？"})
    agent.invoke({"input": "能再说得详细一点吗？"})

if __name__ == "__main__":
    asyncio.run(main3())
   
