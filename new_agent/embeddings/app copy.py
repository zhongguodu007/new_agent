import sqlite3
import chromadb
from chromadb.config import Settings

from fastapi import FastAPI     
from pydantic import BaseModel

from base import EmbeddingModel,TextSpliter

import uvicorn
import os

class TextData(BaseModel):
    text: str
    user_id: int

class AgentDB():
    def __init__(self):
        #初始化Chroma
        self.client = chromadb.PersistentClient(path="db/")
        self.collection = self.client.get_or_create_collection(name="text_embeddings")

        self.sqlite_conn = sqlite3.connect('new_agent.db')
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.sqlite_cursor.execute("PRAGMA foreign_keys = ON;")

        self.init();

    def init(self):
        #用户表
        self.sqlite_cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        #文本数据表
        self.sqlite_cursor.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            text_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """)

        #交互表
        self.sqlite_cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,  -- 用TEXT存储JSON数据
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """)
        self.insert_user("usr1","114514@qq.com","114514")
        self.sqlite_conn.commit()

    #返回插入的用户ID
    def insert_user(self,username,email,password):
        self.sqlite_cursor.execute("""
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?);
        """, (username, email, password))
        self.sqlite_conn.commit()
        return self.sqlite_cursor.lastrowid;  

    #返回插入的文本ID
    def insert_text(self,user_id,text):
        self.sqlite_cursor.execute("""
            INSERT INTO texts (text, user_id)
            VALUES (?, ?);
        """, (text, user_id))
        self.sqlite_conn.commit()
        return self.sqlite_cursor.lastrowid
    
    #插入交互记录
    def insert_log(self,user_id, action, details):
        self.sqlite_cursor.execute("""
            INSERT INTO logs (user_id, action, details)
            VALUES (?, ?, ?);
        """, (user_id, action, details))
        self.sqlite_conn.commit()

    #此处参数embedding是已经由输入的用户文本生成嵌入向量
    def insert_embedding(self,text,embedding,user_id):
        """
        
        """
        text_id = self.insert_text(user_id,text);
        self.collection.add(
            ids=[str(text_id)],#文档ID，这里使用text_id
            embeddings=[embedding],#嵌入向量
            metadatas=[{"user_id": user_id, "text_id": text_id}],#元数据，与向量关联的附加信息（如文本、用户ID、时间戳等）
            documents=[text]
        )
        return text_id;
    
    #查询相似的嵌入向量
    def query_similar_embeddings(self,query_embedding,top_k=5):
        """
        返回最相似的top_k个向量

        返回结果包含以下信息：
        ids 最相似向量的ID列表。
        distances 查询向量与每个相似向量之间的距离（余弦相似度或欧氏距离）
        metadatas 与每个相似向量关联的元数据,如文本ID、用户ID等
        documents 与每个相似向量关联的文本内容
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def close(self):
        self.sqlite_cursor.close()
        self.sqlite_conn.close()
        #self.client.persist()

class AgentApp:
    def __init__(self):
        self.app = FastAPI()
        self.textspliter = TextSpliter();
        self.model = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
        self.db = AgentDB()
        self.register_routes()

    def input_file(self):
        folder_name = "new_agent/file"
        file_names = os.listdir(folder_name)
        file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_name, f))]
        print(file_names)
        for i,f in enumerate(file_names):
            print(os.path.join(folder_name, f))
            splited_doc = self.textspliter.split_text(os.path.join(folder_name, f))
            spliied_embed = self.model.embed_documents(splited_doc)
            for doc,embed in zip(splited_doc,spliied_embed):
                text_id = self.db.insert_embedding(doc,embed,1)

    def register_routes(self):
        @self.app.get("/api/query")
        async def get_similar_texts(query_text: str, top_k: int = 5):

            query_embedding = self.model.embed_query(query_text)
            
            return self.db.query_similar_embeddings(query_embedding,top_k);

    def run(self, host: str = "127.0.0.1", port: int = 8080):
        """运行FastAPI应用"""
        self.input_file()
        uvicorn.run(self.app, host=host, port=port)

    def close(self):
        self.db.close();

if __name__ == "__main__":
    app = AgentApp();
    try:
        app.run();
    except Exception as e:
        print(e)
        app.close();
