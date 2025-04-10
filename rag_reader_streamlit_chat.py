import os
import torch
import logging
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
from smolagents import HfApiModel


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def do_generate(model, tokenizer, retrieved_docs, user_query):
    """
    Génère une réponse à partir des documents récupérés.
    Le prompt est formulé en français et la réponse est générée en markdown.
    """
    # Construction du contexte à partir des documents récupérés
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
Vous êtes un assistant IA.  
Vous répondez à la question.  
Votre réponse doit être informative et concise.  
Vous utilisez le contexte suivant pour générer la réponse.  
Vous répondez UNIQUEMENT en format markdown.
Tu réponds UNIQUEMENT en FRANÇAIS.

Context:
{context}

Question:
{user_query}
    """
    logging.info("Génération du prompt pour la génération de réponse.")

    # Si on utilise un modèle basé sur transformers
    if tokenizer:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info("Réponse générée avec le modèle transformers.")
        return result

    # Sinon, utilisation de HfApiModel
    messages = [{"role": "user", "content": prompt}]
    response = model(messages, stop_sequences=["END"])
    logging.info("Réponse générée avec HfApiModel.")
    return response.content


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Charge et met en cache les modèles et l'index FAISS.
    """
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")
        raise EnvironmentError("HUGGINGFACE_TOKEN n'est pas défini.")
    login(token=hf_token)
    logging.info("Connecté à Hugging Face avec succès.")

    # Chargement du modèle d'embeddings
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info(f"Modèle d'embeddings '{EMBEDDING_MODEL_NAME}' chargé.")

    # Chargement de l'index FAISS
    save_path = "faiss_index_pdf"
    faiss_db = FAISS.load_local(
        save_path,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )
    logging.info("Index FAISS chargé avec succès.")

    # Choix du backend du modèle de langage
    backend = "HfApi"  # Modifier en "transformers" pour utiliser des modèles transformers directement.
    if backend == "transformers":
        MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        logging.info(f"Modèle transformers '{MODEL_NAME}' chargé.")
    else:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        tokenizer = None
        hf_api_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_api_token:
            st.error("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")
            raise EnvironmentError("HUGGINGFACE_TOKEN n'est pas défini.")
        model = HfApiModel(model_id=model_id, token=hf_api_token)
        logging.info(f"Modèle HfApi '{model_id}' chargé.")

    return faiss_db, model, tokenizer


def main():
    st.set_page_config(page_title="EpiRAG", layout="wide")

    # Affichage du logo
    logo_path = "imgs/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        logging.warning("Le fichier de logo n'a pas été trouvé.")

    st.title("EpiRAG")
    st.markdown(
        """
        Cette interface vous permet d'engager une conversation avec un assistant qui utilise un mécanisme de récupération 
        de documents pour fournir des réponses précises et concises.
        """
    )

    # Chargement des modèles et de l'index
    with st.spinner("Chargement des modèles et de l'index..."):
        try:
            faiss_db, model, tokenizer = load_models()
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")
            logging.exception("Erreur lors du chargement des modèles.")
            return

    # Initialisation de l'historique de conversation dans la session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique de conversation
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Zone de saisie pour l'utilisateur
    user_input = st.chat_input("Tapez votre message ici")
    if user_input:
        # Enregistrement et affichage du message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Recherche des documents et génération de la réponse
        with st.spinner("Recherche et génération en cours..."):
            k = 5
            retrieved_docs = faiss_db.similarity_search(query=user_input, k=k)
            logging.info(f"{len(retrieved_docs)} documents récupérés pour la requête.")
            response = do_generate(model, tokenizer, retrieved_docs, user_input)

        # Affichage de la réponse générée
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

        # Optionnel : affichage des documents récupérés dans un expander
        with st.expander("Documents récupérés"):
            for i, doc in enumerate(retrieved_docs):
                source_file = doc.metadata.get("source", None)
                if source_file:
                    source_link = os.path.join("docs", os.path.basename(source_file))
                    st.markdown(f"[📄 Accéder au document {i+1}]({source_link})", unsafe_allow_html=True)
                st.write(doc.page_content)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Une erreur est survenue lors de l'exécution de l'application.")
        st.error(f"Une erreur est survenue : {exc}")
