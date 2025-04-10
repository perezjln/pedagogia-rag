import os
import torch
import logging

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

from smolagents import Tool, CodeAgent, HfApiModel


class FaissRetrievalTool(Tool):
    """
    Tool for retrieving relevant documents from a FAISS index.
    """
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
        logging.info("Performing FAISS similarity search.")
        return faiss_db.similarity_search(query=query, k=k)


class LLMGenerationTool(Tool):
    """
    Tool for generating a response using a language model based on retrieved documents.
    """
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

    def forward(self, retrieved_docs, query: str) -> str:
        # Build context from the retrieved documents.
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        logging.info("Constructed context from retrieved documents.")

        # Format prompt with clear context and question.
        prompt = (
            "You are an AI assistant.\n"
            "Answer the question concisely and informatively using the provided context.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}"
        )

        logging.debug("Generated prompt for LLM generation.")

        # Transformers-based model path.
        if tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            logging.info("Running transformer model generation.")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200)
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            logging.debug("Transformer model output generated.")
            return result

        # HfAPI model invocation.
        logging.info("Running HfApi model generation.")
        messages = [{"role": "user", "content": prompt}]
        response = model(messages, stop_sequences=["END"])
        logging.debug("HfApi model output generated.")
        return response.content


def main():
    # Configure logging: both stream and optional file logging can be set up.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
            # Uncomment the next line to enable file logging
            # logging.FileHandler("app.log")
        ]
    )
    
    logging.info("Starting application.")

    # Load environment variables.
    load_dotenv()
    logging.info("Environment variables loaded.")

    # Check for required tokens.
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        logging.error("HUGGINGFACE_TOKEN is not set in the environment.")
        raise EnvironmentError("HUGGINGFACE_TOKEN is not set in the environment.")
    login(token=hf_token)
    logging.info("Logged into Hugging Face successfully.")

    # Load embedding model.
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")

    # Load FAISS index.
    save_path = "faiss_index_pdf"
    global faiss_db  # Make faiss_db globally available for the retrieval tool.
    faiss_db = FAISS.load_local(
        save_path,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )
    logging.info("FAISS index loaded successfully.")

    # Load language model.
    backend = "HfApi"  # Change to "transformers" to use transformer models directly.
    global tokenizer, model  # Declare as globals so the tools can access them.
    if backend == "transformers":
        MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        logging.info(f"Transformer model '{MODEL_NAME}' loaded.")
    else:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        tokenizer = None  # Transformer tokenizer not used in HfApi mode.
        hf_api_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_api_token:
            logging.error("HUGGINGFACE_TOKEN is not set in the environment.")
            raise EnvironmentError("HUGGINGFACE_TOKEN is not set in the environment.")
        model = HfApiModel(model_id=model_id, token=hf_api_token)
        logging.info(f"HfApi model '{model_id}' loaded.")

    # Instantiate tools.
    generation_tool = LLMGenerationTool()
    retrieval_tool = FaissRetrievalTool()
    logging.info("Tools instantiated successfully.")

    # Create CodeAgent with the tools.
    agent = CodeAgent(tools=[retrieval_tool, generation_tool],
                      model=model,
                      add_base_tools=False)
    logging.info("CodeAgent initialized with tools.")

    # Execute a test query.
    query = (
        "What are the benefits of the attention mechanism in deep learning? "
        "Please cite the retrieved passages in your response."
    )
    logging.info("Running test query through the agent.")
    response = agent.run(query)
    logging.info("Response received from agent.")
    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred during execution:")
