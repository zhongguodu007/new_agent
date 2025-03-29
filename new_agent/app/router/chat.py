from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ..args import (current_agent, 
                    current_mode, 
                    global_memory,
                    init_rag_agent, 
                    init_search_agent,
                    AsyncCallbackManager,
                    AgentExecutorAsyncIteratorCallbackHandler,
                    llm)

router = APIRouter(prefix="/chat", tags=["chat"])

class QueryRequest(BaseModel):
    query_text: str
    usr_name: str



@router.post("/rag")
async def rag_query(request: QueryRequest):
    global mode, current_agent
    if not (mode == "rag"):
        await chat_init_rag_mode()
    return await process_user_input(request.query_text,request.usr_name)


@router.post("/search")
async def search_query(request: QueryRequest):
    global mode, current_agent
    if not (mode == "search"):
        await chat_init_search_mode()
    return await process_user_input(request.query_text,request.usr_name)


@router.get("/get_history")
async def get_history():
    return global_memory.get_all_memory()


async def process_user_input(user_input: str, usr_name: str) -> str:
    """处理用户输入并获取响应"""
    inputs = {
        "input": user_input,
        "chat_history": global_memory.get_all_memory()
    }
    
    response = await current_agent.ainvoke(inputs)
    output = response.get('output', '无响应')
    await global_memory.asave_context(
        inputs={"input": user_input},
        outputs={"output": output}
    )
    if usr_name:
        save_conversation(usr_name)
    return output

async def save_conversation(usr_name: str):
    """保存当前用户的对话到文件"""
    file_path = f"./chat/{usr_name}.json"
    global_memory.save_memory_to_file(file_path)
    return f"对话已保存到 {file_path}"

async def chat_init_rag_mode():
    """本地检索模式"""
    global mode, current_agent
    mode = "rag"
    current_agent = init_rag_agent(
        callback_manager=AsyncCallbackManager([AgentExecutorAsyncIteratorCallbackHandler()]),
        llm=llm,
        persist_directory='./rag',
        collection_name='local_database'
    )

async def chat_init_search_mode():
    """全网搜索模式"""
    global mode, current_agent
    mode = "search"
    current_agent = init_search_agent(
        callback_manager=AsyncCallbackManager([AgentExecutorAsyncIteratorCallbackHandler()]),
        llm=llm
    )
