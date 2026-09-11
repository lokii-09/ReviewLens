import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities
from groq import Groq
import json
import os
import time
import re

print("1. Loading and stratifying reviews across apps...")
df = pd.read_csv('data/raw_reviews.csv')
df = df.dropna(subset=['content'])

app_column = 'app_name' if 'app_name' in df.columns else df.columns[0]
print(f"   Using column '{app_column}' for stratified sampling.")

df_filtered = df[df['content'].str.len() > 80]

# 20 reviews per app = 100 total balanced reviews.
# Groq's free tier gives enough headroom that going a bit above the previous
# 15/app costs almost nothing in time, and gives Louvain more to work with.
samples_per_app = 20
stratified_dfs = []

for app, group in df_filtered.groupby(app_column):
    n_sample = min(len(group), samples_per_app)
    sampled = group.sample(n=n_sample, random_state=42)
    stratified_dfs.append(sampled)
    print(f"   - App '{app}': sampled {n_sample} reviews.")

# Round-robin interleaving (ChatGPT -> Claude -> Copilot -> Gemini -> Perplexity -> ...)
# so a mid-run failure still leaves every app with SOME coverage.
sample_data = []
for i in range(samples_per_app):
    for df_app in stratified_dfs:
        if i < len(df_app):
            row = df_app.iloc[i]
            sample_data.append((row[app_column], row['content']))

print(f"   Total balanced sample size for Graph construction: {len(sample_data)} reviews.\n")

print("2. Connecting to Groq for Scaled Entity Extraction...")
# Free key (no credit card) at https://console.groq.com
# Set it once per terminal session: $env:GROQ_API_KEY="your_key_here"  (PowerShell)
llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BASE_SLEEP = 6.0  # ~10 requests/minute — well under Groq's free-tier 30 RPM cap


def extract_retry_delay(error_str: str) -> float:
    """
    Tries to pull an exact wait time out of the error message if the API
    provides one (some providers do, e.g. 'try again in 12.4s'). Falls back
    to a safe default if we can't parse one out.
    """
    match = re.search(r"try again in (\d+(?:\.\d+)?)", error_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # small buffer on top
    return 20.0  # safe default if we can't parse the actual delay


def extract_entities(app_name: str, text: str):
    """Uses an LLM to extract specific entities and relationships, with retry backoff."""
    prompt = f"""
    You are an AI product data extractor. Extract relationships from the following app review for the app "{app_name}".
    CRITICAL: Avoid overly generic words like "App", "User", "Update", "Data", "Thing", "It" as entities.
    Instead, use highly specific feature names, error types, UI components, or concepts (e.g., "Voice Mode", "Login Screen", "Battery Drain", "Subscription Billing", "Translation Lag").

    Return ONLY a valid JSON list of lists, where each inner list contains exactly three strings:
    ["Source Entity", "RELATIONSHIP_VERB", "Target Entity"].
    Keep entities short (1-3 words).
    Example Output: [["Voice Mode", "CRASHES_ON", "Android Auto"], ["Yearly Subscription", "COSTS", "Too Much"]]

    Review: "{text}"
    """

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            response = llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            cleaned_text = response.choices[0].message.content.strip().replace('```json', '').replace('```', '')
            return json.loads(cleaned_text)  # success — no extra message, we just move on
        except Exception as e:
            error_msg = str(e)
            is_last_attempt = (attempt == max_attempts - 1)

            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if is_last_attempt:
                    print(f"      Rate limit persisted after {max_attempts} attempts — "
                          f"no answer for this review, moving to the next one.")
                    return []
                delay = extract_retry_delay(error_msg)
                print(f"      Rate limit hit (attempt {attempt + 1}/{max_attempts}). Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                print(f"      [!] API Error (attempt {attempt + 1}/{max_attempts}): {error_msg[:200]}")
                if is_last_attempt:
                    print(f"      No answer for this review after {max_attempts} attempts — moving to the next one.")
                    return []
                time.sleep(4.0)
    return []


print(f"3. Building the Knowledge Graph (paced at {BASE_SLEEP}s per call, ~10 req/min)...")
G = nx.Graph()

for i, (app_name, review) in enumerate(sample_data, 1):
    print(f"   [{i}/{len(sample_data)}] Extracting entities for ({app_name})...")
    triplets = extract_entities(app_name, review)

    app_node = app_name.strip().title()
    if app_node not in G:
        G.add_node(app_node, type="APP_ROOT")

    for triplet in triplets:
        if len(triplet) == 3:
            source, relationship, target = triplet
            src, tgt = source.strip().title(), target.strip().title()

            G.add_edge(src, tgt, relation=relationship.upper())
            G.add_edge(app_node, src, relation="HAS_FEATURE_OR_ISSUE")
            G.add_edge(app_node, tgt, relation="RELATES_TO_ISSUE")

    time.sleep(BASE_SLEEP)

print(f"\nGraph built successfully! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

print("4. Running Community Detection (Louvain)...")
communities = list(louvain_communities(G))
print(f"Detected {len(communities)} distinct thematic communities across all apps.")

for i, community in enumerate(communities):
    for node in community:
        G.nodes[node]['community'] = i

print("5. Saving Knowledge Graph and clearing old cache...")
nx.write_graphml(G, "knowledge_graph.graphml")

if os.path.exists("community_summaries.json"):
    os.remove("community_summaries.json")
    print("   Cleared outdated 'community_summaries.json' cache.")

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.4, iterations=60)
colors = [G.nodes[node].get('community', 0) for node in G.nodes()]
node_sizes = [2500 if G.nodes[node].get('type') == 'APP_ROOT' else 400 for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab20,
        node_size=node_sizes, font_size=8, font_weight="bold", edge_color="gray", alpha=0.8)

plt.title("Ecosystem-Wide App-Anchored GraphRAG (Balanced Sample)")
plt.savefig("graph_visualization.png", bbox_inches='tight')
print("\nSuccess! Expanded, app-anchored graph saved to 'knowledge_graph.graphml' and image updated.")