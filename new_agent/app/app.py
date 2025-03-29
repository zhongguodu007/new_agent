from fastapi import FastAPI
from .router import usr,chat,database
import threading
import uvicorn

app = FastAPI()

app.include_router(usr.router,prefix="/usr",tags=["usr"])
app.include_router(chat.router,prefix="/chat", tags=["chat"])
app.include_router(database.router,prefix="/db",tags=["database"])

def run_server(host: str = "127.0.0.1", port: int = 8080):
    """运行FastAPI应用"""
    server_thread = threading.Thread(target=lambda: uvicorn.run(app, host=host, port=port))
    server_thread.daemon = True
    server_thread.start()

if __name__ == "__main__":
    run_server()