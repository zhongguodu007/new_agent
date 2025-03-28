from langchain.callbacks.manager import AsyncCallbackManager
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from langchain_community.chat_models import ChatZhipuAI
from new_agent.agents.all_agents import init_rag_agent, init_search_agent
from new_agent.utils.memory import MemoryChain
from typing import Dict, Any
from fastapi import FastAPI
import os
import json
from local import Agent_DB
import uvicorn

class App:
    def __init__(self):
        self.current_agent = None
        self.mode = None
        self.app = FastAPI()
        self.db = Agent_DB()
        self.llm = ChatZhipuAI(
            model_name="glm-4-0520",
            api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            temperature=1
        )
        self.global_memory = MemoryChain(
            memory_key="chat_history",
            return_messages=True,
            llm=self.llm
        )
        callback_handler = AgentExecutorAsyncIteratorCallbackHandler()
        self.callback_manager = AsyncCallbackManager([callback_handler])
        self.load()
        self.register_routes()

    def register_routes(self):
        @self.app.get("/rag")
        async def rag_query(query_text: str):
            if not(self.mode == "rag"):
                self.init_rag_mode()
            return self.process_user_input(query_text)
        @self.app.get("/search")
        async def search_query(query_text: str):
            if not(self.mode == "search"):
                self.init_search_mode()
            return self.process_user_input(query_text)
        @self.app.post("/usr_login")
        async def usr_login(usr_name: str):
            pass
        @self.app.post("/new_collection")
        async def create_collection(collection_name: str,description:str = "null"):
            return self.db.create_collection(collection_name,description)
        @self.app.post("/delete_collection")
        async def delete_collection(collection_name: str):
            return self.db.delete_collection(collection_name)
        @self.app.post("/add_document")
        async def add_document(collection_name: str,file_path: str,description: str):
            return self.db.add_document(collection_name,file_path,description)
        @self.app.post("/delete_document")
        async def delete_document(collection_name: str,file_path: str):
            return self.db.delete_document(collection_name,file_path)
            
        @self.app.get("/save")#新建集合
        async def save(collection: str,description: str = "default"):
            return self.new_collection(collection,description)
        
    def load(self):
        if(os.path.exists("database/usr.json")):
            with open("database/usr.json",'r') as f:
                self.usr_dict = json.load(f.read())
        else:
            self.usr_dict = {}
    
    async def usr_login(self,usr_name:str,password:str):
        if(self.usr_dict.get(usr_name,-1) == -1):
            return {"status":"用户不存在","code":-1}
        elif(self.usr_dict[usr_name] != password):
            return {"status":"密码错误","code":-2}
        else:
            self.load_conversation("./chat/{usr_name}.json")
            return {"status":"登录成功，已成功加载历史对话","code":1}
    
    async def usr_register(self,usr_name:str,password:str):
        if(self.usr_dict.get(usr_name,-1) != -1):
            return {"status":"该用户名已存在，换一个吧","code":-1}
        else:
            self.usr_dict[usr_name] = password
            return {"status":"注册成功","code":1}

    async def init_rag_mode(self):
        """本地检索模式"""
        self.mode = "rag"
        self.current_agent = init_rag_agent(
            callback_manager=self.callback_manager,
            llm=self.llm,
            persist_directory='./rag',
            collection_name='local_database'
        )

    async def init_search_mode(self):
        """全网搜索模式"""
        self.mode = "search"
        self.current_agent = init_search_agent(
            callback_manager=self.callback_manager,
            llm=self.llm
        )

    async def process_user_input(self,user_input: str,usr_name: str) -> str:
        """处理用户输入并获取响应"""
        inputs = {
            "input": user_input,
            "chat_history": self.global_memory.get_all_memory()
        }
        
        #调用agent
        response = await self.current_agent.ainvoke(inputs)
        output = response.get('output', '无响应')
        await self.global_memory.asave_context(#保存对话
            inputs={"input": user_input},
            outputs={"output": output}
        )
        self.save_conversation(usr_name)
        return output

    async def save_conversation(self,usr_name: str):
        """保存当前用户的对话到文件"""
        file_path = f"./chat/{usr_name}.json"
        self.global_memory.save_memory_to_file(file_path)
        return f"对话已保存到 {file_path}"

    async def load_conversation(self,usr_name: str):
        """从文件加载对话历史"""
        file_path = f"./chat/{usr_name}.json"
        self.global_memory.load_memory_from_file(file_path)
        print(f"已从 {file_path} 加载对话历史")
    
    def run(self, host: str = "127.0.0.1", port: int = 8080):
        """运行FastAPI应用"""
        #self.db.clear()
        print(self.db.doc_dict)
        self.input_file()
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    app = App()
    app.run()