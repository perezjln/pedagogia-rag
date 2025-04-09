import os
import logging
from pathlib import Path
import tqdm
import pandas as pd
from typing import Optional, List
from dotenv import load_dotenv


import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # pip install faiss-cpu ou pip install faiss-gpu
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from markitdown import MarkItDown
from huggingface_hub import login

# Configuration de base du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Séparateurs markdown utilisés pour le découpage du texte
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n",
    "\n\n", "\n", " ", ""
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str],
) -> List[LangchainDocument]:
    """
    Découpe les documents de la base de connaissances en morceaux dont la taille maximale est `chunk_size` tokens.
    """
    logger.info("Initialisation du tokenizer et du text splitter...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    
    # Découpage des documents
    logger.info("Découpage des documents...")
    docs_processed = [
        split for doc in knowledge_base for split in text_splitter.split_documents([doc])
    ]
    
    # Suppression des doublons
    unique_texts = set()
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            docs_processed_unique.append(doc)
    
    logger.info(f"{len(docs_processed_unique)} documents uniques après le traitement.")
    return docs_processed_unique

def do_test_retrieve(knowledge_vector_database):
    """
    Effectue un test de recherche sur le vecteur FAISS.
    """
    logger.info("Lancement du test de recherche ...")
    user_query = "how to create a pipeline object?"
    logger.info(f"Recherche pour la requête : {user_query}")
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=5)
    
    if not retrieved_docs:
        logger.warning("Aucun document trouvé pour la requête.")
        return

    # Affichage des résultats
    logger.info("================================== Document principal ==================================")
    print(retrieved_docs[0].page_content)
    logger.info("================================== Métadonnées ==================================")
    print(retrieved_docs[0].metadata)

def load_documents(documents_path: Path, valid_extensions: List[str] = [".md", ".txt", ".pdf"]) -> List[LangchainDocument]:
    """
    Charge les documents situés dans `documents_path` en filtrant par extension.
    """
    logger.info(f"Chargement des documents depuis : {documents_path}")
    md = MarkItDown()  # Si nécessaire : configurer ou activer les plugins
    documents = []
    
    for file in documents_path.iterdir():
        if file.suffix.lower() in valid_extensions and file.is_file():
            logger.info(f"Traitement du fichier : {file.name}")
            converted = md.convert(str(file))
            # On s'assure que le document a un contenu textuel
            if hasattr(converted, "text_content"):
                documents.append(
                    LangchainDocument(page_content=converted.text_content, metadata={"source": file.name})
                )
            else:
                logger.warning(f"Le fichier {file.name} ne contient pas d'attribut 'text_content'.")
    return documents

if __name__ == "__main__":
    load_dotenv()
    pd.set_option("display.max_colwidth", None)
    
    # Définir le chemin des documents et vérifier son existence
    docs_path = Path("docs")
    if not docs_path.exists() or not docs_path.is_dir():
        logger.error("Le dossier 'docs' n'existe pas. Veuillez vérifier le chemin d'accès.")
        exit(1)
    
    # Chargement de la base de connaissances brute
    RAW_KNOWLEDGE_BASE = load_documents(docs_path)
    
    if not RAW_KNOWLEDGE_BASE:
        logger.error("Aucun document n'a été chargé. Vérifiez le contenu du dossier 'docs'.")
        exit(1)
    
    # Initialisation du modèle d'embeddings
    login(token=os.getenv("HUGGINGFACE_TOKEN"))  # Assurez-vous que le token est défini dans l'environnement
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Longueur maximale de séquence du modèle : {st_model.max_seq_length}")
    
    # Découpage des documents
    docs_processed = split_documents(
        chunk_size=512,
        knowledge_base=RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    
    # Initialisation du modèle d'embeddings avec Langchain
    logger.info("Initialisation du modèle d'embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Pour utiliser la similarité cosinus
    )
    
    # Création de la base de connaissances vectorielle FAISS
    logger.info("Création de l'index FAISS...")
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
    # Sauvegarde locale de l'index FAISS
    save_path = Path("faiss_index_pdf")
    save_path.mkdir(exist_ok=True)
    KNOWLEDGE_VECTOR_DATABASE.save_local(str(save_path))
    logger.info(f"Index FAISS sauvegardé à l'emplacement : {save_path}")
    
    # Test de recherche
    do_test_retrieve(KNOWLEDGE_VECTOR_DATABASE)
