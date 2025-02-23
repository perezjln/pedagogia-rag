from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


## Define a function to generate response using the language model
def do_generate(model, tokenizer, retrieved_docs, user_query):
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
    You are an AI assistant.
    You answer the question.
    You answer should be informative and concise.
    You use the following context to generate the answer.    
    
    Context:
    {context}
    
    Question: 
    {user_query}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response


import os
from huggingface_hub import login
from dotenv import load_dotenv
if __name__ == "__main__":

    # Load the API token from the environment
    load_dotenv()
    login(token=os.environ.get("HF_API_TOKEN"))

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
    faiss_db = FAISS.load_local(save_path, embedding_model, 
                                distance_strategy=DistanceStrategy.COSINE, 
                                allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded from disk")

    print("Load language model ...")
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                                 torch_dtype=torch.float16, 
                                                 device_map="auto")
    
    print("Perform retrieval ...")
    user_query = "What are the benefits of the attention mechanism in deep learning?"
    print(f"\nStarting retrieval for {user_query=}...")
    
    k = 5
    retrieved_docs = faiss_db.similarity_search(query=user_query, k=k)
    
    response = do_generate(model, tokenizer, retrieved_docs, user_query)
    print(response)

