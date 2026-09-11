import networkx as nx
from groq import Groq
import json
import os
import time

print("1. Loading Knowledge Graph...")
G = nx.read_graphml("knowledge_graph.graphml")

# Free key (no credit card) at https://console.groq.com
# Set once per terminal session: $env:GROQ_API_KEY="your_key_here"  (PowerShell)
llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
CACHE_FILE = "community_summaries.json"


def call_llm(prompt: str, json_mode: bool = False, max_attempts: int = 4) -> str:
    """
    Shared, retry-safe wrapper around the Groq chat completion call.
    Used by both the community-summary generation and the final answer
    synthesis, so both get the same rate-limit protection.
    """
    for attempt in range(max_attempts):
        try:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            is_last_attempt = (attempt == max_attempts - 1)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if is_last_attempt:
                    print("   Rate limit persisted after all attempts.")
                    raise
                print(f"   Rate limit hit (attempt {attempt + 1}/{max_attempts}). Waiting 20s...")
                time.sleep(20)
            else:
                print(f"   API Error (attempt {attempt + 1}/{max_attempts}): {error_msg[:200]}")
                if is_last_attempt:
                    raise
                time.sleep(3)
    raise RuntimeError("LLM call failed after all retries")


def generate_community_summaries_batch(graph: nx.Graph) -> dict:
    """
    Groups all communities and asks the LLM to summarize them in a SINGLE batch API call
    to completely bypass per-minute rate limits.
    """
    if os.path.exists(CACHE_FILE):
        print(f"2. Loading cached community summaries from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    communities = {}
    for node, data in graph.nodes(data=True):
        comm_id = str(data.get('community', 0))
        communities.setdefault(comm_id, []).append(node)

    print(f"2. Batching {len(communities)} communities into a SINGLE API call...")

    batch_text = ""
    for comm_id, nodes in communities.items():
        batch_text += f"\n--- COMMUNITY {comm_id} ---\n"
        batch_text += f"ENTITIES: {', '.join(nodes)}\n"
        subgraph = graph.subgraph(nodes)

        edges = []
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('relation', 'RELATED_TO')
            edges.append(f"{u} -> [{rel}] -> {v}")

        batch_text += ("RELATIONSHIPS:\n" + "\n".join(edges) + "\n") if edges else "RELATIONSHIPS: Isolated nodes.\n"

    prompt = f"""You are a senior product analyst. I am providing you with multiple thematic communities extracted from a knowledge graph.
These communities are now structurally linked to specific applications.
For EACH community, write a 1-sentence executive summary of the software issues it describes.
CRITICAL: Explicitly name the specific applications (e.g., ChatGPT, Claude, Gemini, Copilot, Perplexity) involved in the community's issues. Do not use generic terms like "The app".

Respond with ONLY a valid JSON object where the keys are the Community IDs (as strings) and the values are your 1-sentence summaries. No markdown code blocks, no extra text.

GRAPH DATA:
{batch_text}
"""

    raw_output = call_llm(prompt, json_mode=True)
    cleaned_text = raw_output.strip().replace('```json', '').replace('```', '')
    community_reports = json.loads(cleaned_text)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(community_reports, f, indent=2)
    print(f"   Successfully generated and saved summaries to '{CACHE_FILE}'.")

    return community_reports


def answer_global_query(query: str, community_reports: dict) -> str:
    """Synthesize the final answer using the batch-generated community reports."""
    formatted_reports = [f"--- COMMUNITY {comm_id} ---\n{report}" for comm_id, report in community_reports.items()]
    all_reports_text = "\n\n".join(formatted_reports)

    global_prompt = f"""You are an executive AI product strategist. Synthesize a comprehensive answer to the strategic question using the thematic community reports extracted across the entire application ecosystem.

COMMUNITY KNOWLEDGE REPORTS:
{all_reports_text}

USER STRATEGIC QUESTION:
{query}

EXECUTIVE ANSWER:"""

    return call_llm(global_prompt, json_mode=False)


if __name__ == "__main__":
    reports = generate_community_summaries_batch(G)

    global_query = "What are the primary systemic product failures and user frustrations across the entire application ecosystem?"
    print(f"\n3. Asking Global Query: '{global_query}'...\n")

    answer = answer_global_query(global_query, reports)

    print("=" * 60)
    print("=== GLOBAL GRAPHRAG SYNTHESIS ===")
    print("=" * 60)
    print(answer)
    print("=" * 60)