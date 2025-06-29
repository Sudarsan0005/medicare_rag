# 🧠 RAG-based PDF QA System

A Retrieval-Augmented Generation (RAG) based application that allows question-answering over uploaded PDF documents. It uses Langchain, FAISS, NVIDIA models, and Regas evaluation to deliver high-quality, grounded responses.

---

## 🔧 Tech Stack

- **Python**: 3.12  
- **Langchain**: for orchestrating the RAG pipeline  
- **FAISS**: for vector storage and similarity search  
- **PyPDF**: for PDF document parsing  
- **Regas**: for evaluation of RAG responses  
- **NVIDIA NeMo**: for embedding and LLM interaction (optionally supports OpenAI)

---

## 🚀 Installation

Install all dependencies using:

``` text
  -pip install -r requirements.txt
````

---

## 🔐 Environment Variables

Set the following environment variables in your environment:

```env
NVIDIA_API_KEY=""               # Get this from NVIDIA
LLM_MODEL="meta/llama-3.3-70b-instruct"
base_url=""                     # Base URL from NVIDIA model
openai_embedding_model=""             # (Optional) Custom embedding model
OPENAI_API_KEY=""              # (Optional) OpenAI key 
nemo_embedding_model="nvidia/nv-embed-v1"  # 32K context window
```

---

## ▶️ Start the Service

Run the following command to start the service:

```
python main.py
```

---

## 📥 Insert PDF Document

Use the following `curl` command to upload a PDF:

```
curl --location 'http://0.0.0.0:8000/insert_data' \
--form 'file=@"/C:/Users/sudar/Downloads/10050-medicare-and-you_0.pdf"'
```

> ✅ After a successful insertion, **restart the application** to activate the new data for querying.

---

## 💬 Ask Questions (RAG QA)

Send queries to the RAG engine using:

```
curl --location 'http://0.0.0.0:8000/rag_qa' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Your question here"
}'
```

---

## 📊 Evaluation (Regas)

Evaluation was performed using Regas metrics to determine optimal chunking strategy.
Experiments were conducted in `chunks/chunks_evalutor.py`.

### ✅ Best Configuration

* `Chunk Size`: 1200
* `Overlap`: 120

### 📈 Regas Metrics for Best Setting:

```json
{
  "context_recall": 1.0000,
  "faithfulness": 0.8833,
  "factual_correctness(mode=f1)": 0.6020,
  "llm_context_precision_with_reference": 1.0000
}
```

---

## 📌 Notes

* Make sure your API keys and environment settings are valid.
* Restarting the app after PDF upload is essential for chunk processing to take effect.

