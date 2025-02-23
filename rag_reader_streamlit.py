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

    # Transformers
    if tokenizer:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # HfAPI
    messages = [{"role": "user",  "content": prompt}]
    response = model(messages, stop_sequences=["END"])
    return response.content

# Chargement et mise en cache des modèles et index
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
        model = HfApiModel(model_id=model_id, 
                           token=os.environ["HF_API_TOKEN"])
    
    return faiss_db, model, tokenizer

def main():
    st.set_page_config(page_title="Interface RAG", layout="wide")
    st.title("Interface RAG - Retrieval Augmented Generation")
    st.markdown(
        """
        Cette interface vous permet de poser une question, 
        de récupérer les documents pertinents et de générer une réponse en utilisant un modèle de langage.
        """
    )
    
    # Chargement des modèles
    with st.spinner("Chargement des modèles et de l'index..."):
        faiss_db, model, tokenizer = load_models()
    
    # Saisie de la question par l'utilisateur
    user_query = st.text_input("Entrez votre question :", 
                               placeholder="Exemple : Quels sont les avantages du mécanisme d'attention en deep learning ?")
    
    if st.button("Générer la réponse"):
        if user_query.strip() != "":

            # effacer la réponse précédente
            st.empty()

            with st.spinner("Recherche et génération en cours..."):
                # Récupération des documents (k documents les plus pertinents)
                k = 5
                retrieved_docs = faiss_db.similarity_search(query=user_query, k=k)
                                
                # Génération de la réponse
                response = do_generate(model, tokenizer, retrieved_docs, user_query)                

                # Affichage de la réponse
                st.subheader("Réponse générée")
                st.markdown(response)

                # Affichage des documents récupérés
                st.subheader("Documents récupérés")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Document {i+1}"):
                        source_link = doc.metadata.get("source", "#")  # Default to "#" if no link
                        st.markdown(f"[📄 Accéder au document {i+1}]({source_link})", unsafe_allow_html=True)
                        st.write(doc.page_content)

        else:
            st.error("Veuillez entrer une question valide.")

if __name__ == "__main__":
    main()
