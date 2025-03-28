
import os
import sys
import uvicorn 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastapi import FastAPI
from new_agent.app.chatrouter import chatrouter

app = FastAPI()

app.include_router(chatrouter, prefix="/chat", tags=["chat"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 