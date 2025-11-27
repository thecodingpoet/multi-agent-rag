from pathlib import Path

from base_rag_agent import BaseRAGAgent
from dotenv import load_dotenv

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TechAgent(BaseRAGAgent):
    """Tech-specific RAG agent for handling technical support queries."""

    def __init__(self):
        """Initialize the Tech agent with tech-specific paths."""
        docs_path = PROJECT_ROOT / "data" / "tech_docs"
        vector_store_path = PROJECT_ROOT / "vector_stores" / "tech_faiss"

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
        return "Tech"


# Main execution
if __name__ == "__main__":
    print("Initializing Tech RAG Agent...")
    tech_agent = TechAgent()
    tech_agent.initialize()

    test_queries = [
        "How do I reset my email password?",
        "What is the VPN setup process?",
        "How do I request a laptop replacement?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        result = tech_agent.query(query)
        print(f"\nAnswer: {result['answer']}")
