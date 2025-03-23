from new_agent.embeddings.base import EmbeddingModel, TextSpliter
import chromadb
import os
import uuid


def main():
    artics = './其他有意思的论文'
    artical_list = os.listdir(artics)
    splitor= TextSpliter()
    model = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
    db = chromadb.PersistentClient('./rag')
    collection = db.get_collection('rag2')
    # results = collection.get(limit=2, include=["metadatas"])
   
    # print(results) 
    for article in artical_list:
        article_path = os.path.join(artics, article)

        docs = splitor.split_text(article_path)
        embeds = model.embed_documents(docs)
        unique_id = [str(uuid.uuid4()) for _ in range(len(docs))]
        metadatas = [{"paper":article, "pages":i} for i in range(len(docs))]
        assert len(docs) == len(embeds) == len(unique_id) == len(metadatas), (
            "docs、embeds、unique_id、metadatas 的长度必须一致"
        )
        collection.add(documents=docs, embeddings=embeds, ids=unique_id, metadatas=metadatas)
        print(f"成功处理文件: {article}，共 {len(docs)} 个片段")


if __name__ == "__main__":
    main()
