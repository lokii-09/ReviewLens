# ReviewLens 🔍
**Adaptive Multi-Strategy RAG Engine for Cross-Platform App Store Intelligence**

ReviewLens is an end-to-end Retrieval-Augmented Generation (RAG) system engineered to overcome the core limitations of standard naive RAG when analyzing unstructured user feedback across competing AI platforms (**ChatGPT, Claude, Gemini, Copilot, and Perplexity**). 

By decoupling queries using an LLM-powered **Intent Router**, ReviewLens dynamically routes requests to either a high-precision **Hybrid RAG** engine or a macro-level **Hierarchical GraphRAG** engine, serving results via a **FastAPI** backend and an interactive **React + TypeScript** frontend.

---

## 🏛 System Architecture

                          ┌────────────────────────┐
                          │       User Query       │
                          └───────────┬────────────┘
                                      │
                              [ Intent Router ]
                       (Pydantic Schema Validation)
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       [ LOCAL INTENT ]                              [ GLOBAL INTENT ]
  ┌─────────────────────────┐                   ┌─────────────────────────┐
  │     Hybrid Retrieval    │                   │  Hierarchical GraphRAG  │
  │ ─────────────────────── │                   │ ─────────────────────── │
  │ 1. BM25 (Sparse Index)  │                   │ 1. App-Anchored Graph   │
  │ 2. ChromaDB (Dense Vec) │                   │    (Entity Disambig.)   │
  │ 3. Cross-Encoder Rerank │                   │ 2. Louvain Communities  │
  │    (ms-marco-MiniLM)    │                   │ 3. Cluster Summaries    │
  └────────────┬────────────┘                   └────────────┬────────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │  Grounding & Synthesis │
                         │      (Groq LLM)        │
                         └────────────┬────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │  FastAPI / React Client│
                         └────────────────────────┘

---

## ✨ Key Features & Engineering Innovations

### 1. Adaptive Intent Routing
* **Local Scope:** Micro-queries targeting specific features, bug reports, or error patterns (e.g., *"Why does Copilot crash on Android Auto?"*).
* **Global Scope:** Macro-queries requiring dataset-wide aggregation across the ecosystem (e.g., *"What are common cancellation reasons across all apps?"*).

### 2. Two-Stage Hybrid Retrieval (Local Queries)
* **Sparse Search:** BM25 lexical indexing for exact technical error terms and software features.
* **Dense Search:** ChromaDB vector storage embedded with `sentence-transformers/all-MiniLM-L6-v2`.
* **Neural Reranker:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores candidate pools to maximize Context Precision.

### 3. App-Anchored GraphRAG (Global Queries)
* **Entity Disambiguation:** Programmatically binds extracted entity triplets to root `APP_ROOT` nodes (ChatGPT, Gemini, Claude, etc.), preventing generic nodes like *"App"* or *"User"* from collapsing the graph.
* **Community Detection:** Runs the **Louvain algorithm** via NetworkX to detect thematic complaint clusters.
* **Batch Map-Reduce Summaries:** Summarizes each cluster into structured executive reports cached for instant global synthesis.

### 4. Full-Stack Web Application
* **Backend:** FastAPI service exposing `/api/query` and `/api/health` with CORS support and single-instance model caching.
* **Frontend:** Responsive React + TypeScript dashboard built on Vite with real-time query status indicators and dynamic routing engine badges (`LOCAL · Hybrid RAG` vs. `GLOBAL · GraphRAG`).

---

## 📂 Repository Structure

```text
ReviewLens/
├── data/
│   ├── eval_questions.json         # Benchmark dataset across local, comparative, and global queries
│   └── eval_results.csv           # Evaluation results from LLM-as-a-judge
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main chat interface & API client
│   │   ├── App.css                # Dark slate styling & animation system
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── src/
│   ├── api.py                     # FastAPI REST server
│   ├── build_graph.py             # Knowledge Graph extraction & Louvain community builder
│   ├── build_vector_db.py         # Dense vector database indexing pipeline
│   ├── end_to_end.py              # Master orchestrator & intent router
│   ├── query_graph_rag.py         # GraphRAG community summary generation & query engine
│   ├── run_eval.py                # LLM-as-a-judge evaluation harness
│   └── scrape_reviews.py          # Google Play Store review ingestion script
├── community_summaries.json       # Cached Louvain community summaries
├── graph_visualization.png        # Knowledge Graph network visual
├── knowledge_graph.graphml        # Serialized GraphML network
├── requirements.txt
├── .gitignore
└── README.md

🚀 Quickstart Guide
Prerequisites
Python 3.10+

Node.js 18+

Groq API Key (console.groq.com)


1. Backend Setup
# Navigate to project root
cd ReviewLens

# Install Python dependencies
pip install -r requirements.txt

# Set your API Key (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key"

# (Optional) Build Vector DB and Knowledge Graph if starting from scratch
python src/build_vector_db.py
python src/build_graph.py

# Launch FastAPI Server
uvicorn src.api:app --reload --port 8000


2. Frontend Setup
In a new terminal:
cd ReviewLens/frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev

Open http://localhost:5173 in your browser.


📊 Evaluation & Benchmarking
ReviewLens includes an automated evaluation harness (src/run_eval.py) implementing an LLM-as-a-Judge scoring mechanism (1–5 scale):

Faithfulness: Contextual fidelity against hallucination.

Relevance: Direct answer utility for the query.

App Coverage: Objective metric measuring distinct app entity synthesis for ecosystem-wide global queries.