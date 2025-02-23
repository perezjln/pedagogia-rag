import os
import tqdm
import pandas as pd
from typing import Optional, List

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # pip install faiss-cpu ou pip install faiss-gpu
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings


from markitdown import MarkItDown

# Define markdown separators for text splitting
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n",
    "\n\n", "\n", " ", ""
]

# Function to split documents into smaller chunks
def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str],
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    
    # Process documents
    docs_processed = [split for doc in knowledge_base for split in text_splitter.split_documents([doc])]
    
    # Remove duplicates
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)
    
    return docs_processed_unique


def do_test_retrieve():

    print("Perform retrieval ...")
    user_query = "how to create a pipeline object?"
    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    
    # Display results
    print("\n==================================Top document==================================")
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    
    print("Load dataset ...")
    md = MarkItDown() # Set to True to enable plugins

    # list of document in docs/
    ds = [
        {"text": md.convert(f"docs/{file}").text_content, "source": file}
        for file in os.listdir("docs")
    ]

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm.tqdm(ds)
    ]
    
    print("Define embedding model ...")
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    print(f"Model's maximum sequence length: {SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length}")
    
    print("Process documents ...")
    docs_processed = split_documents(
        chunk_size=512,
        knowledge_base=RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    print(f"Number of documents after processing: {len(docs_processed)}")
    
    print("Initialize embedding model ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Use cosine similarity
    )
    
    print("Create FAISS vector database ...")
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
    print("Save local ...")
    save_path = "faiss_index_pdf"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    KNOWLEDGE_VECTOR_DATABASE.save_local(save_path)
    print(f"FAISS index saved at {save_path}")


