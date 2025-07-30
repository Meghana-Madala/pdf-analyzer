"""
# PDF Analyzer with LLM-based Q&A

This is a Streamlit-based chatbot that allows users to upload large PDF files and ask natural language questions about the content. It uses **Azure OpenAI embeddings**, **FAISS vector indexing**, and **GPT-4-mini** to provide human-like, context-aware answers based on the PDF data.

---

## Features

- Upload and process large PDF documents
- Extract and chunk text for semantic search
- Embed text using Azure OpenAI’s `ChatBot` model
- Store and search content using FAISS vector similarity
- Ask natural language questions and receive LLM-generated answers
- Automatically saves/reuses FAISS indexes to speed up repeat queries

---

## Tech Stack

- **Python**
- **Streamlit** – Web interface
- **Azure OpenAI** – Embeddings & GPT-4-mini model
- **FAISS** – Vector similarity search
- **LangChain** – Text chunking
- **PyPDF2** – PDF parsing
- **dotenv** – API key/environment management
- **NumPy** – Numerical operations

---

## Project Structure

main.py               # Main Streamlit app
.env                  # Environment variables
*.faiss               # FAISS index files (auto-generated)
README.md (optional)  # External documentation

---

## Environment Variables

Create a `.env` file in your project root and add the following:

AZURE_OPENAI_API_KEY=your_azure_openai_key  
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

---

## Installation

### 1. Clone the Repository

git clone https://github.com/your-username/pdf-chatbot.git  
cd pdf-chatbot

### 2. Set Up Virtual Environment

python -m venv venv  
source venv/bin/activate        # On Windows: venv\Scripts\activate

### 3. Install Required Packages

pip install -r requirements.txt

### Example Requirements

streamlit  
openai  
langchain  
faiss-cpu  
PyPDF2  
python-dotenv  
numpy

---

## How to Run

streamlit run main.py

Then open your browser to http://localhost:8501/

---

## How It Works

1. Upload PDF
2. Extract and chunk text
3. Generate and store embeddings
4. Ask questions → find relevant chunks → get GPT answer

---

## Author

**Meghana Madala**  
Graduate Student – Computer Science  
University of Memphis


