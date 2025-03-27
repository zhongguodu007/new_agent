from new_agent.embeddings.base import EmbeddingModel, TextSpliter
import chromadb
import os
import uuid


def main():
    artics = './cache'
    artical_list = os.listdir(artics)
    splitor= TextSpliter()
    model = EmbeddingModel(zhipuai_api_key = 'df7f1768a77115a7ffc80e96aad9839b.qAxxUnuN2NLOuFmc', zhipuai_api_base='https://open.bigmodel.cn/api/paas/v4/')
    db = chromadb.PersistentClient('./rag')
    collection = db.get_or_create_collection('local_database')
    collection_name = '与蒸馏相关的论文'
   
    for article in artical_list:
        article_path = os.path.join(artics, article)

        docs = splitor.split_text(article_path)
        embeds = model.embed_documents(docs)
        unique_id = [str(uuid.uuid4()) for _ in range(len(docs))]
        metadatas = [{"doc_id":article, "text_id":i, "collection":collection_name, "description":f'与蒸馏相关的第{article}论文的第{i}个片段'} for i in range(len(docs))]
        assert len(docs) == len(embeds) == len(unique_id) == len(metadatas), (
            "docs、embeds、unique_id、metadatas 的长度必须一致"
        )
        collection.add(documents=docs, embeddings=embeds, ids=unique_id, metadatas=metadatas)
        print(f"成功处理文件: {article}，共 {len(docs)} 个片段")
        print(metadatas)


if __name__ == "__main__":
    main()
