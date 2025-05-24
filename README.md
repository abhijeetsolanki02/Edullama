# Edullama

## 📚 Project Overview

Edullama aims to revolutionize the educational landscape by leveraging the power of local Large Language Models (LLMs) to create personalized, accessible, and privacy-focused learning experiences. Our mission is to bridge the gap between cutting-edge AI and everyday learning, making advanced educational tools available to everyone, regardless of internet connectivity or cloud resource limitations.

## 🛠️ Technologies Used

* **Groq:** The foundational platform for running large language models.
* **LangChain:** Framework for developing applications powered by language models.
* **Streamlit:** For creating intuitive and interactive web interfaces for educational tools.
* **FAISS (or other Vector Store):** For efficient similarity search and retrieval-augmented generation (RAG).
* **`langchain-ollama`:** LangChain integration for Ollama LLMs and Embeddings.
* **`langchain-groq` (Optional):** For integrating fast, cloud-based LLMs for specific tasks where internet access is available and speed is critical.
* **`PyPDFLoader` & `RecursiveCharacterTextSplitter`:** For processing and preparing educational documents.
* **Python:** The primary programming language.

## 🚀 Getting Started

### Prerequisites

* **Python 3.8+**
* **Ollama installed and running:** Download from [ollama.com](https://ollama.com/).
* **Required Ollama Models:**
    * For LLM inference (e.g., Gemma, Llama2, Mistral): `ollama pull gemma2:9b-it` (or your preferred model)
    * For Embeddings: `ollama pull mxbai-embed-large`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Edullama.git](https://github.com/YourUsername/Edullama.git)
    cd Edullama
    ```
    *(Replace `YourUsername/Edullama.git` with your actual repository link)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows: `myenv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file listing all your `pip` dependencies, e.g., `streamlit`, `langchain`, `langchain-community`, `langchain-ollama`, `langchain-groq`, `pypdf`, `python-dotenv`)*

### Running the Application

1.  **Set up environment variables:**
    Create a `.env` file in the root of your project and add your `GROQ_API_KEY` if you are using Groq models.
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

2.  **Start the Streamlit application:**
    ```bash
    streamlit run your_main_app_file.py
    ```
    *(Replace `your_main_app_file.py` with the name of your main Streamlit file, e.g., `pdf_chatbot.py` or `app.py`)*

    Your application should open in your web browser at `http://localhost:8501`.

