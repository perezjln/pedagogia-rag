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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def do_generate(model, tokenizer, retrieved_docs, user_query, student_profile):
    """
    Génère une réponse à partir des documents récupérés, en tenant compte des compétences à développer.
    """
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    skills_to_develop = ", ".join(student_profile["skills_to_develop"])
    prompt = f"""
Vous êtes un assistant IA.  
Vous répondez à la question en français, en format markdown, de manière informative et concise.  
Vous utilisez le contexte suivant et prenez en compte les compétences à développer pour cet étudiant: {skills_to_develop}.

Context:
{context}

Question:
{user_query}
    """
    logging.info("Génération du prompt pour la réponse.")

    if tokenizer:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info("Réponse générée avec transformers.")
        return result

    messages = [{"role": "user", "content": prompt}]
    response = model(messages, stop_sequences=["END"])
    logging.info("Réponse générée avec HfApiModel.")
    return response.content


@st.cache_resource(show_spinner=False)
def load_models():
    """
    Charge les modèles, l'index FAISS et la base de connaissances des profils étudiants.
    """
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACE_TOKEN non défini.")
        raise EnvironmentError("HUGGINGFACE_TOKEN non défini.")
    login(token=hf_token)
    logging.info("Connecté à Hugging Face.")

    # Modèle d'embeddings
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info(f"Modèle d'embeddings '{EMBEDDING_MODEL_NAME}' chargé.")

    # Index FAISS
    save_path = "faiss_index_pdf"
    faiss_db = FAISS.load_local(
        save_path, embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True
    )
    logging.info("Index FAISS chargé.")

    # Modèle de langage
    backend = "HfApi"
    if backend == "transformers":
        MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        logging.info(f"Modèle transformers '{MODEL_NAME}' chargé.")
    else:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        tokenizer = None
        model = HfApiModel(model_id=model_id, token=hf_token)
        logging.info(f"Modèle HfApi '{model_id}' chargé.")

    # Base de connaissances des profils étudiants
    student_profiles = [
        {"name": "Alice", "current_skills": ["Python", "Machine Learning"], "skills_to_develop": ["Deep Learning", "NLP"]},
        {"name": "Bob", "current_skills": ["Java", "Web Development"], "skills_to_develop": ["Cloud Computing", "DevOps"]}
    ]

    return faiss_db, model, tokenizer, student_profiles

def main():
    st.set_page_config(page_title="EpiRAG", layout="wide")

    # Affichage du logo
    logo_path = "imgs/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        logging.warning("Logo non trouvé.")

    st.title("EpiRAG")
    st.markdown("Assistant interactif qui adapte ses réponses aux compétences à développer de l'étudiant.")

    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        try:
            faiss_db, model, tokenizer, student_profiles = load_models()
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            logging.exception("Erreur lors du chargement.")
            return

    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Sélection de l'étudiant
    student_name = st.selectbox("Sélectionnez l'étudiant", [profile["name"] for profile in student_profiles])
    student_profile = next(profile for profile in student_profiles if profile["name"] == student_name)

    # Saisie utilisateur
    user_input = st.chat_input("Tapez votre message ici")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Recherche et génération..."):
            k = 5
            # Conditionnement du RAG avec les compétences à développer
            skills_to_develop = " ".join(student_profile["skills_to_develop"])
            enhanced_query = f"{user_input} {skills_to_develop}"
            retrieved_docs = faiss_db.similarity_search(query=enhanced_query, k=k)
            logging.info(f"{len(retrieved_docs)} documents récupérés.")
            response = do_generate(model, tokenizer, retrieved_docs, user_input, student_profile)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

        # Affichage des documents récupérés
        with st.expander("Documents récupérés"):
            for i, doc in enumerate(retrieved_docs):
                source_file = doc.metadata.get("source", None)
                if source_file:
                    source_link = os.path.join("docs", os.path.basename(source_file))
                    st.markdown(f"[📄 Document {i+1}]({source_link})", unsafe_allow_html=True)
                st.write(doc.page_content)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Erreur lors de l'exécution.")
        st.error(f"Erreur : {exc}")