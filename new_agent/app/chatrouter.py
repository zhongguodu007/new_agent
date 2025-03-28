from fastapi import APIRouter
import asyncio

chatrouter = APIRouter()


@chatrouter.get("/chat")
async def chat(query: str, run_id: str = None):
    """
    聊天接口
    """
    # 模拟异步操作
    await asyncio.sleep(1)
    return {
        "query": query,
        "run_id": run_id,
        "response": "这是一个模拟的聊天响应"
    }