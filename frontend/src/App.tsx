import { useState, type FormEvent } from "react";
import "./App.css";

type QueryResponse = {
  query: string;
  intent: "local" | "global";
  engine: string; 
  answer: string;
};

const EXAMPLE_QUESTIONS = [
  "What do users report about Claude's login issues?",
  "What are the most common reasons users cancel subscriptions across all five apps?",
  "How does battery drain feedback compare between Gemini and Claude?",
];

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data: QueryResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>ReviewLens</h1>
        <p className="tagline">
          Ask about five AI chatbot apps — routed to hybrid retrieval or GraphRAG depending on scope.
        </p>
      </header>

      <form className="query-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What do users say about Claude's login issues?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            className="example-chip"
            onClick={() => setQuestion(q)}
            type="button"
          >
            {q}
          </button>
        ))}
      </div>

      {loading && (
        <div className="status-line">
          <span className="pulse-dot" />
          Routing query and retrieving context…
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {result && !loading && (
        <div className="answer-card fade-in">
          <div className="answer-meta">
            <span className={`badge badge-${result.intent}`}>
              {result.intent === "local" ? "LOCAL" : "GLOBAL"} · {result.engine}
            </span>
          </div>
          <p className="answer-text">{result.answer}</p>
        </div>
      )}

      <footer className="footer">
        System routes local/factual questions to hybrid dense+sparse retrieval with reranking,
        and global/thematic questions to a GraphRAG pipeline over review-derived community summaries.
      </footer>
    </div>
  );
}

export default App;