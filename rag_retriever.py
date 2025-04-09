import logging
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Initialize embedding model ...")
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    
    # Détection du dispositif (GPU si disponible, sinon CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialisation du modèle d'embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # Utilise la similarité cosinus
    )
    
    logger.info("Load local FAISS index ...")
    save_path = "faiss_index_pdf"
    
    # Chargement de l'index FAISS avec gestion d'éventuelles erreurs
    try:
        faiss_db = FAISS.load_local(
            save_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Error loading FAISS index from {save_path}: {e}")
        return
    
    logger.info("FAISS index successfully loaded from disk")
    
    # Définition et lancement de la recherche par similarité
    user_query = "What is the benefits of attention mechanism in deep learning?"
    logger.info(f"Starting retrieval for query: '{user_query}'")
    
    k = 5
    try:
        retrieved_docs = faiss_db.similarity_search(query=user_query, k=k)
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return
    
    # Affichage des résultats
    for idx, doc in enumerate(retrieved_docs):
        print(f"\n\n================================== Document {idx+1} ==================================")
        print(doc.page_content)
        print("================================== Metadata ==================================")
        print(doc.metadata)

import os
from huggingface_hub import login
from dotenv import load_dotenv
if __name__ == "__main__":

    # Load the API token from the environment
    load_dotenv()
    login(token=os.environ.get("HUGGINGFACE_TOKEN"))
    main()
