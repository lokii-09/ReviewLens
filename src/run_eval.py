"""
run_eval.py

Runs the fixed question set through the system(s) and scores every answer
with an LLM judge (faithfulness + relevance, 1-5). Saves a full CSV so you
can manually spot-check the judge's scores, plus prints an aggregate
summary table.

KEY COMPARISON THIS SCRIPT IS BUILT TO SHOW:
  - local/comparative questions -> run through naive (dense-only) AND hybrid
    (BM25 + dense + rerank), so you can show hybrid actually retrieves
    better evidence than naive.
  - global questions -> run through hybrid (forced, even though it's the
    wrong tool) AND GraphRAG, so you can show hybrid genuinely fails to
    summarize across the whole corpus while GraphRAG succeeds. This is the
    central comparison the whole GraphRAG addition is meant to prove.
"""

import json
import csv
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Reuse everything already built and tested in end_to_end.py instead of
# duplicating the retrieval/generation logic here.
from end_to_end import call_llm, hybrid_search, run_global_search, collection

with open("data/eval_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# The five apps in the dataset. Used to compute "coverage" — an objective,
# non-LLM-judged metric for global questions: how many distinct apps does
# the answer actually name? This matters because faithfulness/relevance
# alone can't tell the difference between a genuinely comprehensive answer
# and an honest-but-narrow one (an answer that only saw 5 reviews from 1-2
# apps can still score perfectly on "faithful to its own context" while
# completely failing to synthesize across the ecosystem, which is the
# entire point of a global question).
APP_NAMES = ["ChatGPT", "Gemini", "Claude", "Copilot", "Perplexity"]


def compute_app_coverage(answer: str) -> int:
    """Counts how many of the 5 known apps are explicitly named in the answer."""
    return sum(1 for app in APP_NAMES if app.lower() in answer.lower())

print(f"Loaded {len(questions)} eval questions "
      f"({sum(1 for q in questions if q['category'] == 'local')} local, "
      f"{sum(1 for q in questions if q['category'] == 'comparative')} comparative, "
      f"{sum(1 for q in questions if q['category'] == 'global')} global)\n")


def naive_search(query: str, top_k: int = 5):
    """Dense-only retrieval, no BM25, no reranking — this is the 'before' baseline."""
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


def build_context_str(docs: list) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        blocks.append(f"[{i}] App: {doc['metadata']['app_name']} | Rating: {doc['metadata']['score']}★\nReview: {doc['content']}")
    return "\n\n".join(blocks)


def generate_answer(query: str, context_str: str) -> str:
    prompt = f"""You are an AI product analyst. Answer the user's question strictly using the review snippets provided below. If the answer cannot be determined from the reviews, state that clearly.

--- CONTEXT REVIEWS ---
{context_str}

--- USER QUESTION ---
{query}

--- ANSWER ---"""
    return call_llm(prompt, json_mode=False)


def judge_answer(question: str, context_str: str, answer: str) -> dict:
    """
    LLM-as-judge: scores faithfulness (does the answer stick to the given
    context, no fabrication) and relevance (does it actually address the
    question), each 1-5.
    """
    judge_prompt = f"""You are a strict evaluator of AI-generated answers. Score the ANSWER below on two dimensions, each from 1 to 5:

- faithfulness: does the answer ONLY use information present in the CONTEXT, with no fabricated details? 5 = fully grounded, 1 = mostly made up.
- relevance: does the answer actually address the QUESTION asked? 5 = directly answers it, 1 = off-topic or unhelpful.

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object in exactly this format:
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one short sentence>"}}
"""
    try:
        raw = call_llm(judge_prompt, json_mode=True)
        parsed = json.loads(raw)
        return {
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "relevance": int(parsed.get("relevance", 0)),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"      Judge failed to score: {e}")
        return {"faithfulness": None, "relevance": None, "reasoning": f"judge_error: {e}"}


csv_path = "data/eval_results.csv"
fieldnames = ["id", "category", "engine", "question", "answer",
              "faithfulness", "relevance", "judge_reasoning", "app_coverage"]

# RESUME SUPPORT: if a previous run got interrupted (e.g. daily quota hit),
# don't redo work that already succeeded — that would waste more of an
# already-limited daily token budget on questions we already have answers for.
already_done = set()
existing_rows = []
if os.path.exists(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing_rows.append(row)
            already_done.add((row["id"], row["engine"]))
    print(f"Resuming: found {len(already_done)} already-completed (question, engine) pairs in {csv_path}\n")

results = list(existing_rows)

# Open in write mode once, write existing rows back, then append new ones
# as they complete — so a crash partway through this run still saves everything up to that point.
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(existing_rows)
csv_file.flush()

for i, q in enumerate(questions, 1):
    qid, category, question_text = q["id"], q["category"], q["question"]

    if category in ("local", "comparative"):
        engines_to_run = ["naive", "hybrid"]
    else:
        engines_to_run = ["hybrid_forced", "graph"]

    # Skip engines already completed for this question
    engines_to_run = [e for e in engines_to_run if (qid, e) not in already_done]
    if not engines_to_run:
        print(f"[{i}/{len(questions)}] {qid} ({category}): already complete, skipping.")
        continue

    print(f"[{i}/{len(questions)}] {qid} ({category}): {question_text}")

    for engine in engines_to_run:
        print(f"   -> engine: {engine}")

        if engine == "naive":
            docs = naive_search(question_text)
            context_str = build_context_str(docs)
            answer = generate_answer(question_text, context_str)

        elif engine == "hybrid":
            docs = hybrid_search(question_text, final_top_k=5, candidate_pool_size=25)
            context_str = build_context_str(docs)
            answer = generate_answer(question_text, context_str)

        elif engine == "hybrid_forced":
            # Deliberately using hybrid retrieval on a GLOBAL question it's not
            # built for, to show it can't summarize across the whole corpus.
            docs = hybrid_search(question_text, final_top_k=5, candidate_pool_size=25)
            context_str = build_context_str(docs)
            answer = generate_answer(question_text, context_str)

        elif engine == "graph":
            with open("community_summaries.json", "r", encoding="utf-8") as f:
                community_reports = json.load(f)
            context_str = "\n\n".join(f"Community {k}: {v}" for k, v in community_reports.items())
            answer = run_global_search(question_text)

        scores = judge_answer(question_text, context_str, answer)
        coverage = compute_app_coverage(answer) if category == "global" else None

        row = {
            "id": qid,
            "category": category,
            "engine": engine,
            "question": question_text,
            "answer": answer,
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "judge_reasoning": scores["reasoning"],
            "app_coverage": coverage,
        }
        results.append(row)
        writer.writerow(row)
        csv_file.flush()  # save immediately — don't lose progress if the next call hits the quota wall

        time.sleep(3)  # stay comfortably under Groq's free-tier rate limit

csv_file.close()
print(f"\nFull results saved to {csv_path} — open it and spot-check ~15-20 rows manually.")

# --- Aggregate summary ---
print("\n" + "=" * 70)
print("AGGREGATE SUMMARY (mean scores by category + engine)")
print("=" * 70)

summary = {}
for r in results:
    key = (r["category"], r["engine"])
    if key not in summary:
        summary[key] = {"faithfulness": [], "relevance": [], "app_coverage": []}
    if r["faithfulness"] is not None:
        summary[key]["faithfulness"].append(r["faithfulness"])
    if r["relevance"] is not None:
        summary[key]["relevance"].append(r["relevance"])
    if r["app_coverage"] is not None:
        summary[key]["app_coverage"].append(r["app_coverage"])

for (category, engine), scores in sorted(summary.items()):
    avg_f = sum(scores["faithfulness"]) / len(scores["faithfulness"]) if scores["faithfulness"] else 0
    avg_r = sum(scores["relevance"]) / len(scores["relevance"]) if scores["relevance"] else 0
    n = len(scores["faithfulness"])
    line = f"{category:12s} | {engine:15s} | n={n:2d} | avg faithfulness={avg_f:.2f} | avg relevance={avg_r:.2f}"
    if scores["app_coverage"]:
        avg_c = sum(scores["app_coverage"]) / len(scores["app_coverage"])
        line += f" | avg app_coverage={avg_c:.1f}/5"
    print(line)

print("\nDone. NOTE: faithfulness/relevance alone tend to look similar across")
print("engines on global questions — an honest-but-narrow answer still scores")
print("well on both. app_coverage is the metric that actually shows whether")
print("an answer synthesized across the whole ecosystem or just a few reviews;")
print("that's the number that justifies building GraphRAG in the first place.")