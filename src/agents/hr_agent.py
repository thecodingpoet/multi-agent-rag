from pathlib import Path

from base_rag_agent import BaseRAGAgent
from dotenv import load_dotenv

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class HRAgent(BaseRAGAgent):
    """HR-specific RAG agent for handling HR-related queries."""

    def __init__(self):
        """Initialize the HR agent with HR-specific paths."""
        docs_path = PROJECT_ROOT / "data" / "hr_docs"
        vector_store_path = PROJECT_ROOT / "vector_stores" / "hr_faiss"

        super().__init__(
            docs_path=docs_path,
            vector_store_path=vector_store_path,
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o-mini",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=4,
        )

    def get_agent_name(self) -> str:
        """Return the name of this agent."""
        return "HR"


if __name__ == "__main__":
    hr_agent = HRAgent()
    hr_agent.initialize()

    test_queries = [
        "How many vacation days do employees get?",
        "What is the remote work policy?",
        "How does the performance review process work?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        result = hr_agent.query(query)
        print(f"\nAnswer: {result['answer']}")
