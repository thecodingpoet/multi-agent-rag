from pathlib import Path

from dotenv import load_dotenv

from agents.base_rag_agent import BaseRAGAgent

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
