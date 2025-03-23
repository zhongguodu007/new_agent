import chromadb
from langchain_chroma import Chroma
from new_agent.embeddings.base import EmbeddingModel, TextSpliter
from new_agent.retrival.base import NewRetriever
import uuid
from typing import List
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import asyncio

if __name__ == '__main__':
    chroma_store = Chroma(
        collection_name="rag2",
        persist_directory="./rag"
    )

    retreval = NewRetriever(
        store=chroma_store,
        search_kwargs={"n_results":2}
    )
    llm = ChatOpenAI(
        model_name="glm-4-0520",
        api_key="df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # 或者选择其他适合的类型，如"map_reduce", "refine", "map_rerank"
    retriever=retreval,
    return_source_documents=True  # 如果需要返回源文档的话
    )

    query = "文章的题目"

    result = qa_chain(query)
    print(result)
   
