import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import json
import sys
sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from new_agent.embeddings.base import EmbeddingModel,TextSpliter

class Agent_DB:
    def __init__(self,file_folder):
        self.textspliter = TextSpliter();
        self.embeddingmodel = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
        # 初始化客户端
        self.client = chromadb.PersistentClient(path="./rag")
        self.collection = self.client.get_or_create_collection(name="main")
        self.file_folder = file_folder
        self.load()
    
    def load(self):
        """加载数据库"""
        if(os.path.exists("database/collection.json")):
            with open("database/collection.json","r") as f:
                self.collection_dict = json.load(f)[0]
        else:
            self.collection_dict = {}
        if(os.path.exists("database/document.json")):
            with open("database/document.json","r") as f:
                self.document_dict = json.load(f)[0]
        else:
            self.document_dict = {}
        if(os.path.exists("database/usr.json")):
            with open("database/usr.json","r") as f:
                self.usr_dict = json.load(f)[0]
        else:
            self.usr_dict = {}

    def dump(self):
        with open("database/collection.json","w") as f:
            f.write(json.dumps([self.collection_dict]))
        with open("database/document.json","w") as f:
            f.write(json.dumps([self.document_dict]))
        with open("database/usr.json","w") as f:
            f.write(json.dumps([self.usr_dict]))

    def get_collection(self):
        return {"status":"查询成功",
                "collection":self.collection_dict}
    
    def get_document(self,collection_name):
        file_list = self.document_dict[collection_name]
        return {"status":"查询成功",
                "document":file_list}

    def new_collection(self,collection_name:str,description:str):
        if(collection_name in list(self.collection_dict.keys())):
            return {"status":"不能创建同名集合"}
        self.collection_dict[collection_name] = {
            "description":description,
        }
        self.document_dict[collection_name] = []
        self.dump()
        return {"status":"新集合创建成功"}

    def delete_collection(self,collection_name):
        if not(collection_name in list(self.collection_dict.keys())):
            return {"status":"集合不存在"}
        #del self.collection_name[self.collection_name.index(collection_name)]
        del self.collection_dict[collection_name]
        del self.document_dict[collection_name]
        self.collection.delete(
            where={"collection": collection_name}
        )
        self.dump()
        return {"status":"集合已成功删除"}

    def delete_document(self,collection_name,file_name):
        if not (collection_name in list(self.collection_dict.keys())):
            return {"status":"集合不存在"}
        if not self.check_document_repeat(collection_name,file_name):
            return {"status":"文章不存在"};
        del self.document_dict[collection_name][self.document_dict[collection_name].index(file_name)]
        self.collection.delete(
            where={"doc_id":file_name}
        )
        self.dump()
        return {"status":f"文章已成功从集合{collection_name}中删除"};

    def check_document_repeat(self,collection_name,file_name):
        file_list = self.document_dict[collection_name]
        if(file_name in file_list):
            return True
        return False

    def add_document(self,collection_name,file_name,description):
        if not (collection_name in list(self.collection_dict.keys())):
            return {"status":"集合不存在"}
        if self.check_document_repeat(collection_name,file_name):
            return {"status":"不能添加同名文章"};
        file_path = self.file_folder+file_name
        if not os.path.exists(file_path):
            return {"status":f"文件不存在，或文件未放入{self.file_folder}文件夹下"}
        
        self.document_dict[collection_name].append(file_name)
        self.dump()
        #print("{}  {}".format(doc_id,file_name))
        
        splited_doc = self.textspliter.split_text(file_path)
        spliied_embed = self.embeddingmodel.embed_documents(splited_doc)
        for txt_id,(doc,embed) in enumerate(zip(splited_doc,spliied_embed)):
            self.collection.add(
                ids=[f"{collection_name}-{file_name}-{txt_id}"],#文档ID
                embeddings=[embed],#嵌入向量
                metadatas=[{"collection":collection_name,
                            "description":description,
                            "doc_id": file_name,
                            "text_id": txt_id}],#元数据，与向量关联的附加信息（如文本、用户ID、时间戳等）
                documents=[doc]
            )
        os.remove(file_path)
        return {"status":f"文章已成功添加至集合{collection_name}中"};

    
if __name__ == "__main__":
    db = Agent_DB()
