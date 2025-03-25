import chromadb
import json
from fastapi import FastAPI

from base import EmbeddingModel,TextSpliter

import uvicorn
import os
import requests

class AgentDB():
    def __init__(self):
        #初始化Chroma
        self.client = chromadb.PersistentClient(path="db/")
        self.collection = self.client.get_or_create_collection(name="text_embeddings")
        self.load();
    
    def clear(self):
        self.doc_dict.clear()
        self.doc_dict["doc_id"] = self.doc_id = 1
        self.doc_dict["text_id"] = self.text_id = 1
        self.dump()
        self.collection.delete(ids=self.collection.get()["ids"])
    
    def insert_document(self,file_path):
        self.doc_dict[self.doc_id] = {"file_path":file_path}
        self.doc_id+=1;
        self.doc_dict["doc_id"] = self.doc_id
        self.dump()
        return self.doc_id-1
    
    def check_document_repeat(self,file_path):
        for f in  self.doc_dict.values():
            if type(f) == int: 
                continue
            if f["file_path"] == file_path:
                return True
        return False
    
    def load(self):
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                self.doc_dict = json.load(f)
            self.doc_id = self.doc_dict["doc_id"]
            self.text_id = self.doc_dict["text_id"]
        except FileNotFoundError as e:
            self.doc_dict = {"doc_id":1,"text_id":1}
            self.doc_id = self.text_id = 1;
    
    def dump(self):
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_dict, f, ensure_ascii=False, indent=4)

    #此处参数embedding是已经由输入的用户文本生成嵌入向量
    def insert_embedding(self,text,embedding,doc_id,text_id):
        self.collection.add(
            ids=[str(self.text_id)],#文档ID，这里使用text_id
            embeddings=[embedding],#嵌入向量
            metadatas=[{"doc_id": doc_id, "text_id": text_id}],#元数据，与向量关联的附加信息（如文本、用户ID、时间戳等）
            documents=[text]
        )
        self.text_id += 1;
        self.doc_dict["text_id"] = self.text_id
        self.dump()
    
    #查询相似的嵌入向量
    def query_similar_embeddings(self,query_embedding,top_k=5):
        with open("a.json",'w',encoding="utf-8") as f:
            json.dump(query_embedding,f)
        """
        返回最相似的top_k个向量

        返回结果包含以下信息：
        ids 最相似向量的ID列表。
        distances 查询向量与每个相似向量之间的距离（余弦相似度或欧氏距离）
        metadatas 与每个相似向量关联的元数据，文章id，段id
        documents 与每个相似向量关联的文本内容
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances","documents","metadatas"]#,"embeddings"]
        )
        #with open("b.json",'w',encoding="utf-8") as f:
            #for i in results["embeddings"]:
                #pass#json.dump(i.tolist(),f)
        return results#["documents","metadatas"]
    
    def delete_document(self,doc_id):
        del self.doc_dict[doc_id]
        self.collection.delete(where={"doc_id": doc_id})

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
        for f in file_names:
            file_path = os.path.join(folder_name, f)
            self.input_single(file_path)

    def input_single(self,file_path):
        if(self.db.check_document_repeat(file_path)):
            return None
        doc_id = self.db.insert_document(file_path)
        print("{}  {}".format(doc_id,file_path))
        splited_doc = self.textspliter.split_text(file_path)
        spliied_embed = self.model.embed_documents(splited_doc)
        for txt_id,(doc,embed) in enumerate(zip(splited_doc,spliied_embed)):
            self.db.insert_embedding(doc,embed,doc_id,txt_id+1)

    def register_routes(self):
        """
        发送post请求
        url = "http://127.0.0.1:8080/api/data"
        data = {
            "action":"insert",
            "file_path":"new_agent/file/114514.pdf"
        }
        response = requests.post(url, json=data)
        data = {
            "action":"delete",
            "doc_id":"114514"
        }
        response = requests.post(url, json=data)
        
        发送get请求
        url = "http://127.0.0.1:8080/api/query"
        params = {
            "query_text": "computer",
            "top_k": 5
        }
        response = requests.get(url, params=params)
        或输入url
        http://127.0.0.1:8080/api/query?query=computer&top_k=5
        """
        @self.app.post("/api/data")
        async def process_text(data):
            if(data["action"] == "insert"):
                self.db.insert_document(data["file_path"])
                self.input_single(data["file_path"])
                return {"message": "Document inserted"}
            
            elif(data["action"] == "delete"):
                self.db.delete_document(data["doc_id"])
                return {"message": "Document deleted"}
        @self.app.get("/api/query")
        async def query_texts(query_text: str, top_k: int = 5):
            query_embedding = self.model.embed_query(query_text)

            return self.db.query_similar_embeddings(query_embedding,top_k);

    def run(self, host: str = "127.0.0.1", port: int = 8080):
        """运行FastAPI应用"""
        #self.db.clear()
        print(self.db.doc_dict)
        self.input_file()
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    app = AgentApp();
    app.run();
