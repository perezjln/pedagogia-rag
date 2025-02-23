import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv
from smolagents import HfApiModel

# Fonction de génération de réponse à partir des documents récupérés
def do_generate(model, tokenizer, retrieved_docs, user_query):
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
You are an AI assistant.
You answer the question.
Your answer should be informative and concise.
You use the following context to generate the answer.
You OnLY answer with markdown format.

Context:
{context}

Question: 
{user_query}
    """
    # Si le modèle utilise transformers
    if tokenizer:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Si le modèle utilise HfApiModel
    messages = [{"role": "user", "content": prompt}]
    response = model(messages, stop_sequences=["END"])
    return response.content

# Chargement et mise en cache des modèles et de l'index
@st.cache_resource(show_spinner=False)
def load_models():

    load_dotenv()
    hf_token = os.environ.get("HF_API_TOKEN")
    if hf_token is None:
        st.error("La variable d'environnement HF_API_TOKEN n'est pas définie.")
    login(token=hf_token)
    
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    save_path = "faiss_index_pdf"
    faiss_db = FAISS.load_local(
        save_path, 
        embedding_model, 
        distance_strategy=DistanceStrategy.COSINE, 
        allow_dangerous_deserialization=True
    )
    
    backend = "HfApi"
    if backend == "transformers":
        MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        tokenizer = None
        model = HfApiModel(model_id=model_id, token=os.environ["HF_API_TOKEN"])
    
    return faiss_db, model, tokenizer

def main():

    st.set_page_config(page_title="EpiRAG", layout="wide")

    # Chemin vers le fichier logo
    logo_path = "imgs/logo.png"
    st.image(logo_path, width=150)  # Vous pouvez ajuster la largeur selon vos besoins

    st.title("EpiRAG")
    st.markdown(
        """
        Cette interface vous permet d'engager une conversation avec un assistant qui utilise un mécanisme de récupération 
        de documents pour fournir des réponses précises et concises.
        """
    )
    
    # Chargement des modèles et de l'index
    with st.spinner("Chargement des modèles et de l'index..."):
        faiss_db, model, tokenizer = load_models()
    
    # Initialisation de l'historique de conversation dans session_state
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
        # Afficher le message de l'utilisateur dans l'interface de chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Recherche des documents pertinents et génération de la réponse
        with st.spinner("Recherche et génération en cours..."):
            k = 5
            retrieved_docs = faiss_db.similarity_search(query=user_input, k=k)
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
    main()
