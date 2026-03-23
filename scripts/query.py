from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

try:
    from swarm import Agent, Swarm
except ImportError as exc:
    raise ImportError(
        "OpenAI Swarm is not installed in this environment. "
        "The PyPI package 'swarm' is a different project. "
        "Install the OpenAI framework with: "
        "pip install git+https://github.com/openai/swarm.git"
    ) from exc


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "faiss_index"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local literature RAG index with a Swarm router agent."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing the FAISS index. Default: {DEFAULT_INDEX_DIR}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Embedding model used when loading the FAISS index. "
            f"Default: {DEFAULT_EMBEDDING_MODEL}"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM used by all Swarm agents. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of chunks to retrieve for each literature search. Default: 6",
    )
    parser.add_argument(
        "--question",
        help="Ask a single question from the command line instead of starting interactive mode.",
    )
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def create_openai_client() -> OpenAI:
    load_env_file(DEFAULT_ENV_PATH)

    base_url = os.environ.get("PROXY_BASE_URL")
    if not base_url:
        raise ValueError("PROXY_BASE_URL is not set. Add it to your .env file.")

    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )


def load_vectorstore(index_dir: Path, embedding_model: str) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory does not exist: {index_dir}. "
            "Run scripts/indexer.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_search_literature_tool(vectorstore: FAISS, default_top_k: int):
    def search_literature(query: str, top_k: int = default_top_k) -> str:
        """Search the indexed course literature for evidence relevant to the user's question.

        Use this before answering. Prefer exact evidence from the retrieved passages and keep
        the final answer grounded in the returned source and page citations.
        """

        try:
            search_size = max(1, int(top_k))
        except (TypeError, ValueError):
            search_size = default_top_k

        results = vectorstore.similarity_search(query, k=search_size)

        if not results:
            return "No relevant literature chunks were found."

        formatted_chunks: list[str] = []
        for index, doc in enumerate(results, start=1):
            metadata = doc.metadata
            file_name = metadata.get("file_name", Path(metadata.get("source", "unknown")).name)
            page = metadata.get("page", "unknown")
            excerpt = " ".join(doc.page_content.split())
            formatted_chunks.append(
                f"[{index}] Source: {file_name} | Page: {page}\nExcerpt: {excerpt}"
            )

        return "\n\n".join(formatted_chunks)

    return search_literature


def extract_final_answer(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"].strip()
    return "No answer was generated."


def build_agents(model: str, search_literature):
    methods_agent = Agent(
        name="Methods Analyst",
        model=model,
        instructions=(
            "You are the Methods Analyst for a scientific literature assistant. "
            "Answer only questions about methods, experimental setup, datasets, baselines, "
            "training procedures, evaluation protocols, and metrics. "
            "Always call search_literature before answering. "
            "Ground every claim in the retrieved evidence. "
            "Cite every substantive point with document name and page number in this format: "
            "(Document, p. X). "
            "If the retrieved evidence is insufficient, say that clearly instead of guessing."
        ),
        functions=[search_literature],
    )

    results_agent = Agent(
        name="Results Extractor",
        model=model,
        instructions=(
            "You are the Results Extractor for a scientific literature assistant. "
            "Answer only questions about results, numbers, quantitative findings, ablations, "
            "benchmarks, comparisons, and reported performance. "
            "Always call search_literature before answering. "
            "Prioritize exact numbers and direct comparisons when available. "
            "Cite every substantive point with document name and page number in this format: "
            "(Document, p. X). "
            "If the evidence does not contain the requested result, state that explicitly."
        ),
        functions=[search_literature],
    )

    reviewer_agent = Agent(
        name="Skeptical Reviewer",
        model=model,
        instructions=(
            "You are the Skeptical Reviewer for a scientific literature assistant. "
            "Answer only questions about limitations, threats to validity, assumptions, "
            "weaknesses, critique, missing comparisons, and possible failure modes. "
            "Always call search_literature before answering. "
            "Base criticism on the retrieved papers rather than inventing objections. "
            "Cite every substantive point with document name and page number in this format: "
            "(Document, p. X). "
            "If a limitation is your inference, label it explicitly as an inference."
        ),
        functions=[search_literature],
    )

    general_agent = Agent(
        name="General Synthesizer",
        model=model,
        instructions=(
            "You are the General Synthesizer for a scientific literature assistant. "
            "Handle broader or mixed questions that are not primarily about methods, results, "
            "or limitations. "
            "Always call search_literature before answering. "
            "Synthesize across the retrieved evidence while staying faithful to the sources. "
            "Cite every substantive point with document name and page number in this format: "
            "(Document, p. X). "
            "If the papers do not support a conclusion, say so clearly."
        ),
        functions=[search_literature],
    )

    def transfer_to_methods_analyst():
        """Use for questions about methods, setup, datasets, architectures, protocols, or metrics."""

        return methods_agent

    def transfer_to_results_extractor():
        """Use for questions about results, numbers, ablations, comparisons, or performance tables."""

        return results_agent

    def transfer_to_skeptical_reviewer():
        """Use for questions about limitations, critique, threats to validity, assumptions, or weaknesses."""

        return reviewer_agent

    def transfer_to_general_synthesizer():
        """Use for broad, mixed, summary, or otherwise uncategorized literature questions."""

        return general_agent

    router_agent = Agent(
        name="Router Agent",
        model=model,
        instructions=(
            "You are the router agent for a scientific literature assistant. "
            "Your only job is to choose the best specialist for the user's question by calling "
            "exactly one handoff function. Do not answer the question yourself. "
            "Use Methods Analyst for methods, setup, dataset, and metrics questions. "
            "Use Results Extractor for results, numbers, ablations, and comparisons. "
            "Use Skeptical Reviewer for limitations and critique. "
            "Use General Synthesizer for everything else."
        ),
        functions=[
            transfer_to_methods_analyst,
            transfer_to_results_extractor,
            transfer_to_skeptical_reviewer,
            transfer_to_general_synthesizer,
        ],
    )

    return router_agent


def ask_question(client: Swarm, router_agent: Agent, question: str) -> None:
    response = client.run(
        agent=router_agent,
        messages=[{"role": "user", "content": question}],
    )
    agent_name = response.agent.name if response.agent else "Unknown Agent"
    answer = extract_final_answer(response.messages)

    print(f"\nAgent: {agent_name}")
    print(answer)


def interactive_loop(client: Swarm, router_agent: Agent) -> None:
    print("Literature assistant ready. Type a question or 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        ask_question(client, router_agent, question)


def main() -> None:
    args = parse_args()
    vectorstore = load_vectorstore(
        index_dir=args.index_dir.expanduser().resolve(),
        embedding_model=args.embedding_model,
    )
    search_literature = build_search_literature_tool(vectorstore, args.top_k)
    router_agent = build_agents(args.model, search_literature)
    swarm_client = Swarm(client=create_openai_client())

    if args.question:
        ask_question(swarm_client, router_agent, args.question)
        return

    interactive_loop(swarm_client, router_agent)


if __name__ == "__main__":
    main()
