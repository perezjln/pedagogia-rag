# pedagogia-rag

pedagogia-rag is a Retrieval-Augmented Generation (RAG) framework built using popular NLP libraries such as LangChain, Transformers, Sentence Transformers, and FAISS. It demonstrates several approaches for indexing, retrieving, and generating responses from a knowledge base. The project includes scripts for processing various document types (Markdown, text, PDF) and interactive interfaces using Streamlit and smolagents.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Indexing Documents](#indexing-documents)
  - [Generating Responses](#generating-responses)
  - [Interactive Interfaces](#interactive-interfaces)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)

## Overview

This project exemplifies a full pipeline for RAG applications, where unstructured data (documents, PDFs, markdown files) is:
1. **Processed and Split** into smaller, tokenized chunks using a HuggingFace tokenizer and a recursive text splitter.
2. **Embedded** with a pretrained model (e.g., `thenlper/gte-small`) and stored locally in a FAISS vector index.
3. **Queried** using similarity search to retrieve relevant document passages.
4. **Augmented** with an answer generated using a language model (e.g., `meta-llama/Llama-3.2-3B-Instruct` or using the HF API).

The design supports a range of applications and interfaces, including command-line tools and interactive web apps built with Streamlit.

## Project Structure

The repository is organized into several files, each focusing on a different aspect of RAG:

- **rag_index.py**  
  Processes a dataset from Hugging Face, splits documents into manageable chunks, computes embeddings, creates a FAISS vector database, and performs a test retrieval.

- **rag_index_pdf.py**  
  Extends the indexing script by loading documents (including PDFs) from a specified directory, using a markdown converter (MarkItDown) for preprocessing, and then building a FAISS index.

- **rag_reader.py**  
  Loads the FAISS vector database, retrieves relevant documents for a user query, and generates a response using a causal language model.

- **rag_reader_smolagents.py**  
  Demonstrates integration with smolagents by wrapping retrieval and response generation in custom tools and using a CodeAgent to handle queries.

- **rag_reader_streamlit.py**  
  Provides a Streamlit-based interactive interface where users can enter a query, see retrieved documents, and view a generated response.

- **rag_reader_streamlit_chat.py**  
  Implements a conversational chat interface with Streamlit, preserving conversation history, and providing a chat-like experience for dialogue-based interactions.

- **rag_reader_streamlit_chat_studentprofile.py**  
  Similar to `rag_reader_streamlit_chat.py`, but tailored for a specific use case involving student profiles, allowing users to query and interact with the system in a more personalized manner.

- **rag_retriever.py**  
  A standalone script focused solely on retrieving documents from the local FAISS index based on a query, useful for testing and debugging the retrieval component.

- **requierements.txt**  
  Lists all Python dependencies required for the project, ensuring reproducibility and an easy setup process.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/pedagogia-rag.git
   cd pedagogia-rag
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requierements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your tokens:
   
   ```env
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   HF_API_TOKEN=your_hf_api_token_here
   ```

## Usage

### Indexing Documents

- **Using the Hugging Face Dataset (rag_index.py):**

  ```bash
  python rag_index.py
  ```

  This script downloads a predefined dataset (`m-ric/huggingface_doc`), processes the documents, splits them into chunks, computes embeddings, creates a FAISS index, and performs a test retrieval.

- **Processing Local Documents and PDFs (rag_index_pdf.py):**

  Ensure you have a folder named `docs` with your Markdown, Text, or PDF files. Then run:

  ```bash
  python rag_index_pdf.py
  ```

  This version uses the `MarkItDown` converter to extract text from PDFs and other document formats.

### Generating Responses

- **Command Line Response Generation (rag_reader.py):**

  ```bash
  python rag_reader.py
  ```

  This script loads the locally saved FAISS index, retrieves documents based on a user query, and generates an answer using a language model.

- **Using smolagents (rag_reader_smolagents.py):**

  ```bash
  python rag_reader_smolagents.py
  ```

  This script wraps both retrieval and answer generation in agents to handle queries dynamically.

### Interactive Interfaces

- **Streamlit Interface (rag_reader_streamlit.py):**

  Launch the Streamlit app:

  ```bash
  streamlit run rag_reader_streamlit.py
  ```

  This opens a web interface where you can input queries and see both generated responses and the source documents.

- **Streamlit Chat Interface (rag_reader_streamlit_chat.py):**

  Start the chat-based interface with:

  ```bash
  streamlit run rag_reader_streamlit_chat.py
  ```

  Enjoy a conversational experience that maintains context and provides retrieved document details.

## Environment Variables

For the scripts to authenticate and access Hugging Face services, ensure the following variables are set:

- **HUGGINGFACE_TOKEN**: Required for loading model weights and dataset access.
- **HF_API_TOKEN**: Required for API-based language model access and token-based operations in some scripts.

These can be set in your operating system’s environment or stored in a `.env` file at the project root.

## Requirements

The project relies on several Python packages. Key dependencies include:

- `torch>=1.13`
- `transformers>=4.30.0`
- `sentence-transformers>=2.2.2`
- `langchain>=0.1.0`
- `langchain-community>=0.0.21`
- `langchain-huggingface>=0.0.6`
- `faiss-cpu`
- `markitdown[all]`
- `pandas>=1.5.0`
- `tqdm>=4.64.0`

Make sure all dependencies are installed using the provided `requierements.txt`.
