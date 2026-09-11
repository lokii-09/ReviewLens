import os
import json
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

print("=" * 60)
print("1. Initializing LLM Client & Data Indexes...")
print("=" * 60)

# Free key (no credit card) at https://console.groq.com
# Set once per terminal session: $env:GROQ_API_KEY="your_key_here"  (PowerShell)
llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- LOAD DATA FOR BM25 & LOOKUPS ---
df = pd.read_csv('data/raw_reviews.csv')
df = df.dropna(subset=['content'])
df = df[df['content'].str.len() > 15]

reviews_list = df['content'].tolist()
ids_list = [f"{row['app_name']}_{row['review_id']}" for _, row in df.iterrows()]
metadata_list = [{'app_name': row['app_name'], 'score': row['score']} for _, row in df.iterrows()]

tokenized_reviews = [doc.lower().split() for doc in reviews_list]
bm25 = BM25Okapi(tokenized_reviews)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="app_reviews", embedding_function=sentence_transformer_ef)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Local search indexes & models loaded successfully.\n")


import re

def parse_retry_seconds(error_str: str) -> float:
    """
    Correctly parses Groq's 'try again in 1m16.896s' style messages.
    The previous version only captured the first number it found, which
    misread '1m16.896s' as 1 second instead of 76 seconds — this fixes that.
    """
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 2.0  # small buffer
    return 20.0  # fallback if we can't parse it


def call_llm(prompt: str, json_mode: bool = False, max_attempts: int = 4) -> str:
    """
    Shared, retry-safe wrapper around the Groq chat completion call.
    Every LLM call in this file (routing, local answer, global answer)
    goes through this, so all of them get the same rate-limit protection —
    important once the eval script calls this dozens of times in a row.

    Using llama-3.1-8b-instant here: its free-tier daily token budget
    (500K TPD) is far larger than llama-3.3-70b-versatile's (100K TPD),
    which is what we were actually running into — a DAILY cap, not a
    per-minute one, so short waits never fixed it.
    """
    for attempt in range(max_attempts):
        try:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            is_last_attempt = (attempt == max_attempts - 1)

            if "tokens per day" in error_msg.lower() or "TPD" in error_msg:
                # Daily quota exhausted — retrying at all is pointless until
                # the actual reset window Groq tells us about. Fail fast and
                # loud instead of burning more of an already-exhausted budget.
                wait = parse_retry_seconds(error_msg)
                print(f"   DAILY TOKEN QUOTA EXHAUSTED. Groq says try again in {wait:.0f}s "
                      f"({wait/60:.1f} min). Stopping retries — re-run the script after that.")
                raise

            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if is_last_attempt:
                    print("   Rate limit persisted after all attempts.")
                    raise
                delay = parse_retry_seconds(error_msg)
                print(f"   Rate limit hit (attempt {attempt + 1}/{max_attempts}). Waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"   API Error (attempt {attempt + 1}/{max_attempts}): {error_msg[:200]}")
                if is_last_attempt:
                    raise
                time.sleep(3)
    raise RuntimeError("LLM call failed after all retries")


# =====================================================================
# SECTION 1: INTENT ROUTING
# =====================================================================
class QueryIntent(BaseModel):
    """
    Validated locally after the LLM responds, since Groq's JSON mode
    guarantees valid JSON but not this exact schema/field the way
    Gemini's response_schema did — so we check it ourselves.
    """
    query_type: Literal["local", "global"] = Field(
        description="'local' for specific apps/features/bugs, 'global' for high-level cross-app trends."
    )


def classify_query(query: str) -> str:
    """Classifies user intent, with a safe fallback if the model ever returns something unexpected."""
    prompt = f"""Classify the intent of this user query as either "local" or "global".
local = asking about specific apps, features, or exact bugs.
global = asking for high-level summaries, dataset-wide trends, or systemic issues across apps.

Respond with ONLY a JSON object in exactly this format: {{"query_type": "local"}} or {{"query_type": "global"}}

Query: "{query}"
"""
    raw_output = call_llm(prompt, json_mode=True)
    try:
        parsed = QueryIntent(**json.loads(raw_output))
        return parsed.query_type
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"   Router returned an unexpected shape ({e}); defaulting to 'local'.")
        return "local"  # local is the safer default — it won't try to summarize a whole ecosystem incorrectly


# =====================================================================
# SECTION 2: LOCAL SEARCH ENGINE (HYBRID BM25 + CHROMADB + RERANK)
# =====================================================================
def hybrid_search(query: str, final_top_k: int = 5, candidate_pool_size: int = 25):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool_size]

    chroma_results = collection.query(query_texts=[query], n_results=candidate_pool_size)
    chroma_ids = chroma_results['ids'][0]

    unique_candidates = {}
    for idx in bm25_top_indices:
        doc_id = ids_list[idx]
        unique_candidates[doc_id] = {"content": reviews_list[idx], "metadata": metadata_list[idx]}

    for i, doc_id in enumerate(chroma_ids):
        if doc_id not in unique_candidates:
            unique_candidates[doc_id] = {
                "content": chroma_results['documents'][0][i],
                "metadata": chroma_results['metadatas'][0][i],
            }

    candidate_docs = list(unique_candidates.values())

    cross_inp = [[query, doc["content"]] for doc in candidate_docs]
    cross_scores = cross_encoder.predict(cross_inp)
    for idx in range(len(cross_scores)):
        candidate_docs[idx]["cross_score"] = float(cross_scores[idx])

    candidate_docs = sorted(candidate_docs, key=lambda x: x["cross_score"], reverse=True)
    return candidate_docs[:final_top_k]


def run_local_search(query: str) -> str:
    print("   [ENGINE] Running Hybrid RAG (ChromaDB + BM25 + Cross-Encoder)...")
    top_reviews = hybrid_search(query, final_top_k=5, candidate_pool_size=25)

    context_blocks = []
    for i, doc in enumerate(top_reviews, 1):
        context_blocks.append(
            f"[{i}] App: {doc['metadata']['app_name']} | Rating: {doc['metadata']['score']}★ "
            f"(Rerank Score: {doc['cross_score']:.2f})\nReview: {doc['content']}"
        )
    context_str = "\n\n".join(context_blocks)

    rag_prompt = f"""You are an AI product analyst. Answer the user's question strictly using the review snippets provided below. If the answer cannot be determined from the reviews, state that clearly.

--- CONTEXT REVIEWS ---
{context_str}

--- USER QUESTION ---
{query}

--- ANSWER ---"""

    return call_llm(rag_prompt, json_mode=False)


# =====================================================================
# SECTION 3: GLOBAL SEARCH ENGINE (HIERARCHICAL GRAPHRAG)
# =====================================================================
def run_global_search(query: str) -> str:
    print("   [ENGINE] Running Hierarchical GraphRAG (Community Summaries)...")

    cache_file = "community_summaries.json"
    if not os.path.exists(cache_file):
        return "Error: 'community_summaries.json' not found. Run 'python src/query_graph_rag.py' once to generate it."

    with open(cache_file, "r", encoding="utf-8") as f:
        community_reports = json.load(f)

    formatted_reports = [f"--- COMMUNITY CLUSTER {comm_id} ---\n{report}" for comm_id, report in community_reports.items()]
    all_reports_text = "\n\n".join(formatted_reports)

    global_prompt = f"""You are an executive AI product strategist. Answer the user's high-level strategic question by synthesizing the thematic community reports extracted across the entire application ecosystem.

COMMUNITY KNOWLEDGE REPORTS:
{all_reports_text}

USER STRATEGIC QUESTION:
{query}

EXECUTIVE ANSWER:"""

    return call_llm(global_prompt, json_mode=False)


# =====================================================================
# SECTION 4: MASTER ORCHESTRATOR
# =====================================================================
def process_user_query(query: str) -> dict:
    """
    Returns a structured result (not just prints) — the eval script we build
    next needs the classified intent and answer text programmatically, not
    just as console output.
    """
    print("\n" + "=" * 60)
    print(f"USER QUERY: '{query}'")
    print("=" * 60)

    print("Step 1: Routing query intent...")
    intent = classify_query(query)
    print(f"   [ROUTER DECISION] -> Classified as: {intent.upper()}")

    print("\nStep 2: Executing Search Engine...")
    if intent == "local":
        final_answer = run_local_search(query)
    else:
        final_answer = run_global_search(query)

    print("\n" + "=" * 60)
    print("=== FINAL GROUNDED SYSTEM OUTPUT ===")
    print("=" * 60)
    print(final_answer)
    print("=" * 60 + "\n")

    return {"query": query, "intent": intent, "answer": final_answer}


if __name__ == "__main__":
    test_local = "Why does Copilot's Live Voice Mode crash on Android Auto over Bluetooth?"
    process_user_query(test_local)

    test_global = "What are the most common reasons users cancel subscriptions across all 5 AI applications?"
    process_user_query(test_global)