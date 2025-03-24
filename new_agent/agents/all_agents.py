from langchain.callbacks.manager import AsyncCallbackManager
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_chroma import Chroma
from new_agent.retrival.base import NewRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA

def create_retrieval_tool(persist_directory: str, collection_name: str)->Tool:
    
    chroma_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    retriever = NewRetriever(
            store=chroma_store,
            search_kwargs={"n_results":8}
        )
    def retrieve_documents(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        formatted = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            del metadata['distance']
            formatted.append(f"元数据：{metadata}\n内容：{content}")
        return "\n---\n".join(formatted)
    
    return Tool(
        name=f"{collection_name}知识库",
        func=retrieve_documents,
        description=f"从{collection_name}知识库中检索相关信息，返回格式化后的文档内容和元数据。"
    )


def init_rag_agent(callback_manager:AsyncCallbackManager, llm: ChatZhipuAI, 
                   persist_directory: str = './rag', collection_name: list[str]=['rag2'], 
                   ) -> AgentExecutor:
    qa_tools = []
    if len(collection_name) == 1:
        collection_name = collection_name[0]
        retrieval_tool = create_retrieval_tool(persist_directory, collection_name)
        qa_tools = [retrieval_tool]

    else:
        for collection in collection_name:
            retrieval_tool = create_retrieval_tool(persist_directory, collection)
            qa_tools.append(retrieval_tool)

    prompt = ChatPromptTemplate.from_template(
        """
        你正在以一个智能体的身份运行。你拥有以下工具：
        {tools}
        此外，以下是之前的对话历史：
        {chat_history}

        使用这些工具来回答用户的问题：“{input}”。你之前已经进行了以下思考和行动：
        {agent_scratchpad}

        请严格从以下工具中选择一个执行，如果有多个知识库请一次检索每个知识库获取相关答案保证完整性：
        工具列表：{tool_names}

        输出格式必须包含：
        Thought: [你的思考过程]
        Action: [必须是工具列表中的名称（如：本地知识库]
        Action Input: [传递给工具的参数]

        如果问题已解决，请直接输出：
        Final Answer: [最终答案]
        """
    )
    agent = create_react_agent(
        llm=llm,
        tools=qa_tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=qa_tools,
        handle_parsing_errors=True, 
        callback_manager=callback_manager,
        verbose=True
    )
    return agent_executor

def init_search_agent(callback_manager:AsyncCallbackManager, llm: ChatZhipuAI) -> AgentExecutor:
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
        Action: [必须是工具列表中的名称（如：]
        Action Input: [传递给工具的参数]

        如果问题已解决，请直接输出：
        Final Answer: [最终答案]
        """
    )
    tools = []
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True, 
        callback_manager=callback_manager,
        verbose=True
    )
    return agent_executor