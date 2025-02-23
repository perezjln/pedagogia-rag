
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings


if __name__ == "__main__":

    print("Initialize embedding model ...")
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Use cosine similarity
    )

    print("Load local FAISS ...")
    save_path = "faiss_index_pdf"
    faiss_db = FAISS.load_local(save_path, embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded from disk")

    print("Perform retrieval ...")
    #user_query = "how to create a pipeline object?"
    user_query = "What is the benefits of attention mechanism in deep learning?"

    print(f"\nStarting retrieval for {user_query=}...")

    k = 5
    retrieved_docs = faiss_db.similarity_search(query=user_query, k=k)
    
    # Display results
    for idx in range(k):
        print("\n\n==================================document==================================")
        print(retrieved_docs[idx].page_content)
        print("==================================Metadata==================================")
        print(retrieved_docs[idx].metadata)
