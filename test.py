from langchain_chroma import Chroma
from new_agent.retrival.base import NewRetriever
from langchain.chains import RetrievalQA
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.manager import AsyncCallbackManager
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent
from langchain_core.runnables import RunnableSequence

async def main():
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

    prompt = ChatPromptTemplate.from_template(
        """
        你正在以一个智能体的身份运行。你拥有以下工具：
        {tools}

        使用这些工具来回答用户的问题：“{input}”。你之前已经进行了以下思考和行动：
        {agent_scratchpad}

        请严格从以下工具中选择一个执行：
        工具列表：{tool_names}

        输出格式必须包含：
        Thought: [你的思考过程]
        Action: [必须是工具列表中的名称（如：本地知识库 或 DuckDuckGo Search）]
        Action Input: [传递给工具的参数]

        如果问题已解决，请直接输出：
        Final Answer: [最终答案]
        """
    )

    agent = create_react_agent(
        llm=llm,
        tools=[retrieval_tool, search_tool],
        prompt=prompt
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[retrieval_tool, search_tool],
        handle_parsing_errors=True,  # 自动处理解析错误
        #max_iterations=10,           # 最大尝试次数
        callback_manager=AsyncCallbackManager(
            [AgentExecutorAsyncIteratorCallbackHandler()]
        ),
        verbose=True
    )

    #question = "我想知道OBJECT-ORIENTED RELATIONAL DISTILLATION FOR OBJECT DETECTION这篇文章的主要内容，这是在知识库的文章，用中文回答我的问题"
    question = "我想知道OBJECT-ORIENTED RELATIONAL DISTILLATION FOR OBJECT DETECTION这篇文章的具体方法是什么，这是在知识库的文章，用中文回答我的问题"
    result = await agent_executor.ainvoke({"input": question})
    print("最终答案：", result.get("output"))

if __name__ == "__main__":
    asyncio.run(main())
# if __name__ == '__main__':
#     chroma_store = Chroma(
#         collection_name="rag2",
#         persist_directory="./rag"
#     )

#     retreval = NewRetriever(
#         store=chroma_store,
#         search_kwargs={"n_results":8}
#     )
#     llm = ChatOpenAI(
#         model_name="glm-4-0520",
#         api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
#         openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
#     )
#     prompt_template = ChatPromptTemplate.from_template(
#     "使用以下上下文回答最后的问题。如果你不知道答案，请直接说你不知道，不要尝试编造答案。\n\n上下文:\n{context}\n\n问题: {question}"
#     )
#     qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff",  # 或者选择其他适合的类型，如"map_reduce", "refine", "map_rerank"
#     retriever=retreval,
#     return_source_documents=False,
#     chain_type_kwargs={"prompt":prompt_template}
#     )

#     query = "我想知道A Novel Contrastive Signal Generative Framework for Accurate Graph Learning这篇文章的主要内容"

#     result = qa_chain(query)
#     print(result)
   
