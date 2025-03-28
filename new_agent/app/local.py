import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import json
from new_agent.embeddings.base import EmbeddingModel,TextSpliter

class Agent_DB:
    def __init__(self):
        self.textspliter = TextSpliter();
        self.embeddingmodel = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
        # 初始化客户端
        self.client = chromadb.PersistentClient(path="./rag")
        self.collection = self.client.get_or_create_collection(name="main")
        self.load()
    
    def load(self):
        """加载数据库"""
        if(os.path.exists("database/collection.json")):
            with open("database/collection.json","r") as f:
                self.collection_name = json.load(f.read())
        else:
            self.collection_name = []
        if(os.path.exists("database/document.json")):
            with open("database/doc_id.json","r") as f:
                self.collection_dict = json.load(f.read())
        else:
            self.collection_dict = {}
    
    def dump(self):
        with open("database/collection","w") as f:
            f.write(json.dump(self.collection_name))
        with open("database/doc_id.json","w") as f:
            f.write(json.dump(self.collection_dict))

    def create_collection(self,collection_name:str,description:str = "null"):
        if(collection_name in self.collection_name):
            return {"status":"不能创建同名集合"}
        self.collection_dict[collection_name] = {
            "description":description,
            "document_cnt":0,
            "document":[]
        }
        self.collection_name.append(collection_name);
        self.dump()
        return {"status":"新集合创建成功"}

    def delete_collection(self,collection_name):
        if not(collection_name in self.collection_name):
            return {"status":"集合不存在"}
        del self.collection_name[self.collection_name.index(collection_name)]
        del self.collection_dict[collection_name]
        self.collection.delete(
            where={"collection": collection_name}
        )
        self.dump()
        return {"status":"集合已成功删除"}

    def delete_document(self,collection_name,file_path):
        if not (collection_name in self.collection_name):
            return {"status":"集合不存在"}
        if not self.check_document_repeat(collection_name,file_path):
            return {"status":"文章不存在"};
        del self.collection_dict[collection_name]["document"][self.collection_dict[collection_name]["document"].index(file_path)]
        self.collection.delete(
            where={"file_path":file_path}
        )
        self.dump()
        return {"status":f"文章已成功从集合{collection_name}中删除"};

    def check_document_repeat(self,collection_name,file_path):
        file_list = self.collection_dict[collection_name]["document"]
        if(file_path in file_list):
            return True
        return False

    def add_document(self,collection_name,file_path,description):
        if not (collection_name in self.collection_name):
            return {"status":"集合不存在"}
        if self.check_document_repeat(collection_name,file_path):
            return {"status":"不能添加同名文章"};
        
        self.collection_dict[collection_name]["document"].append(file_path)
        doc_id = self.collection_dict[collection_name]["document_cnt"]
        self.collection_dict[collection_name]["document_cnt"]+=1
        self.dump()
        #print("{}  {}".format(doc_id,file_path))
        splited_doc = self.textspliter.split_text(file_path)
        spliied_embed = self.embeddingmodel.embed_documents(splited_doc)
        for txt_id,(doc,embed) in enumerate(zip(splited_doc,spliied_embed)):
            self.collection.add(
                ids=[str(doc_id*10000+txt_id)],#文档ID，这里使用text_id
                embeddings=[embed],#嵌入向量
                metadatas=[{"collection":collection_name,
                            "description":description,
                            "file_path":file_path,
                            "doc_id": doc_id,
                            "text_id": txt_id}],#元数据，与向量关联的附加信息（如文本、用户ID、时间戳等）
                documents=[doc]
            )
        return {"status":f"文章已成功添加至集合{collection_name}中"};

    