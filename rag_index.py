import os
import logging
from pathlib import Path
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

import datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define markdown separators for text splitting
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    ""
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str],
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens using a HuggingFace tokenizer.
    
    Parameters:
        chunk_size (int): Maximum number of tokens in each chunk.
        knowledge_base (List[LangchainDocument]): List of documents to split.
        tokenizer_name (Optional[str]): Name of the pretrained model tokenizer.
    
    Returns:
        List[LangchainDocument]: Processed list of document chunks with duplicates removed.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    
    # Process and split each document in the knowledge base
    docs_processed = []
    for doc in knowledge_base:
        splits = text_splitter.split_documents([doc])
        docs_processed.extend(splits)
    
    # Remove duplicate chunks (based on their content)
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)
    
    return docs_processed_unique


def do_test_retrieve(knowledge_db: FAISS) -> None:
    """
    Perform a test retrieval on the provided FAISS vector database.
    
    Parameters:
        knowledge_db (FAISS): The FAISS vector database instance.
    """
    user_query = "how to create a pipeline object?"
    logging.info(f"Starting retrieval for query: '{user_query}'")
    
    retrieved_docs = knowledge_db.similarity_search(query=user_query, k=5)
    
    if retrieved_docs:
        logging.info("==================================Top document==================================")
        logging.info(retrieved_docs[0].page_content)
        logging.info("==================================Metadata==================================")
        logging.info(retrieved_docs[0].metadata)
    else:
        logging.info("No documents retrieved.")


def main() -> None:
    pd.set_option("display.max_colwidth", None)
    
    logging.info("Loading dataset ...")
    try:
        ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    raw_knowledge_base: List[LangchainDocument] = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds, desc="Loading documents")
    ]
    
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    logging.info("Defining embedding model ...")
    # Display model's maximum sequence length
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info(f"Model's maximum sequence length: {model.max_seq_length}")
    
    logging.info("Splitting and processing documents ...")
    docs_processed = split_documents(
        chunk_size=512,
        knowledge_base=raw_knowledge_base,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    logging.info(f"Number of documents after processing: {len(docs_processed)}")
    
    logging.info("Initializing embedding model ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Use cosine similarity
    )
    
    logging.info("Creating FAISS vector database ...")
    knowledge_vector_database = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
    save_path = Path("faiss_index")
    logging.info("Saving FAISS vector database locally ...")
    save_path.mkdir(exist_ok=True)
    knowledge_vector_database.save_local(str(save_path))
    logging.info(f"FAISS index saved at {save_path.resolve()}")
    
    # Optionally run a test retrieval query
    do_test_retrieve(knowledge_vector_database)


if __name__ == "__main__":
    main()
