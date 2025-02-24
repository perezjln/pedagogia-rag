import os
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

from smolagents import Tool, CodeAgent, HfApiModel

class FaissRetrievalTool(Tool):
    name = "faiss_retriever"
    description = "Retrieve relevant documents from FAISS index."
    inputs = {
        "query": {
            "type": "string",
            "description": "The user query to search relevant documents."
        },
        "k": {
            "type": "integer",
            "description": "Number of documents to retrieve.",
            "nullable": True
        }
    }
    output_type = "array"

    def forward(self, query: str, k: int = 5):
        return faiss_db.similarity_search(query=query, k=k)

class LLMGenerationTool(Tool):
    name = "llm_generator"
    description = "Generate a response using a language model based on retrieved documents."
    inputs = {
        "retrieved_docs": {
            "type": "array",
            "description": "List of retrieved documents from FAISS."
        },
        "query": {
            "type": "string",
            "description": "User query for which a response should be generated."
        }
    }
    output_type = "string"

    def forward(self, retrieved_docs, query: str):

        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f'''
        You are an AI assistant.
        You answer the question.
        Your answer should be informative and concise.
        You use the following context to generate the answer.

        Context:
        {context}

        Question:
        {query}
        '''

        # Transformers
        if tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=200)
            return tokenizer.decode(output[0], skip_special_tokens=True)

        # HfAPI
        messages = [{"role": "user",  "content": prompt}]
        response = model(messages, stop_sequences=["END"])
        return response.content

if __name__ == "__main__":

    # Charger les variables d'environnement
    load_dotenv()
    login(token=os.environ.get("HF_API_TOKEN"))

    # Charger le modèle d'embedding
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Charger l'index FAISS
    save_path = "faiss_index_pdf"
    faiss_db = FAISS.load_local(
        save_path, embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )

    # Charger le modèle de langage
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

    # Instancier les outils
    generation_tool = LLMGenerationTool()
    retrieval_tool = FaissRetrievalTool()

    # Créer l'agent CodeAgent avec les outils
    agent = CodeAgent(tools=[retrieval_tool, generation_tool], 
                      model=model, 
                      add_base_tools=False)

    # Exécuter une requête test
    query = "What are the benefits of the attention mechanism in deep learning? You cite the retrieved passages in your response."
    response = agent.run(query)
    print(response)
