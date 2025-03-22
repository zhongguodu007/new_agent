import chromadb

if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path='./rag')
    collection = chroma_client.get_collection(name='my_rag')

    collection.add(
        documents=[
            "这是一个关于水果的文件",
            "这是一个关于蔬菜的文件"
        ],
        ids=["1","2"],
        embeddings=[[1,1,1],[1,1,2]],
    )


    results = collection.query(query_texts=["这是关于苹果的文件"],query_embeddings=[1,1,3],n_results=1)
    print(results)
