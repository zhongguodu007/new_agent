from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Dict, List, Optional, Any
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_chroma import Chroma
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from new_agent.embeddings.base import EmbeddingModel

class NewRetriever(BaseRetriever):

    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    store: Chroma = None
    collection: List[str] = []

    embeddingmodel: EmbeddingModel = None


    class Config:

        extra = 'forbid'
        allow_population_by_field_name = True

    def __init__(self, **data):
        BaseRetriever.__init__(self)
        self.search_kwargs = data.get("search_kwargs",{'n_results': 8})
        self.store = data["store"]
        self.collection = data.get("collection",[])
        self.embeddingmodel = EmbeddingModel(
            zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc',
              zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')

    
    def get_relevant_documents(self, query: str, **kwargs) ->List[Document]:

        merge_args = {**kwargs, **self.search_kwargs}

        query_embed = asyncio.run(self.embeddingmodel.aembed_query(query))

        if len(self.collection) > 0:
            filter_condition = {'collection':{"$in":self.collection}}
            results = self.store._collection.query(query_texts=[query], query_embeddings=[query_embed], where=filter_condition, **merge_args)
        else:
            results = self.store._collection.query(query_texts=[query], query_embeddings=[query_embed], **merge_args)

        document = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        docs = []
        for doc_txt, metadata, distance in zip(document, metadatas, distances):
            
            metadata['distance'] = distances
            doc = Document(
                page_content=doc_txt,
                metadata=metadata
            )
            docs.append(doc)

        return docs
        
    

if __name__ == "__main__":
    print(111)


    
