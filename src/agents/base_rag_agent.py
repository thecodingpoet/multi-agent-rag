import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseRAGAgent(ABC):
    """Base class for RAG agents with common functionality."""

    def __init__(
        self,
        docs_path: Path,
        vector_store_path: Path,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retrieval_k: int = 4,
    ):
        """
        Initialize the RAG agent.

        Args:
            docs_path: Path to the documents directory
            vector_store_path: Path to save/load the vector store
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of documents to retrieve
        """
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store: Optional[FAISS] = None
        self.agent = None
        self.retriever = None
        self.logger = logging.getLogger(f"agents.{self.get_agent_name().lower()}")

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the name of this agent (e.g., 'HR', 'Finance', 'Tech').
        Used for logging and identification.
        """
        pass

    def load_documents(self):
        """Load all markdown files from the documents directory."""
        loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
        )
        documents = loader.load()
        return documents

    def split_documents(self, documents):
        """Split documents into chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,  # Track chunk position in original document
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_vector_store(self, chunks):
        """Create FAISS vector store from document chunks."""
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store

    def save_vector_store(self):
        """Save FAISS vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        Path(self.vector_store_path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_path))
        self.logger.debug(f"Vector store saved to {self.vector_store_path}")

    def load_vector_store(self) -> bool:
        """
        Load FAISS vector store from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if Path(self.vector_store_path).exists():
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.debug(f"Vector store loaded from {self.vector_store_path}")
            return True
        return False

    def build_agent(self) -> None:
        """
        Build a RAG agent with a retrieval tool.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        model = init_chat_model(self.llm_model, model_provider="openai", temperature=0)

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.retrieval_k}
        )

        agent_name = self.get_agent_name()

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information from the knowledge base to help answer questions."""
            retrieved_docs = self.retriever.invoke(query)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        system_prompt = (
            f"You are a helpful {agent_name} assistant. "
            f"Use the retrieve_context tool to search the {agent_name} knowledge base "
            "when you need information to answer questions. "
            "Always cite your sources when providing information from the knowledge base."
        )

        self.agent = create_agent(
            model, tools=[retrieve_context], system_prompt=system_prompt
        )

    def initialize(self):
        """Initialize the RAG agent by loading or creating vector store."""
        agent_name = self.get_agent_name()

        self.logger.info(f"Initializing {agent_name} agent...")

        loaded = self.load_vector_store()

        if not loaded:
            self.logger.info("Creating new vector store...")
            documents = self.load_documents()
            self.logger.debug(f"Loaded {len(documents)} documents")

            chunks = self.split_documents(documents)
            self.logger.debug(f"Split into {len(chunks)} chunks")

            self.vector_store = self.create_vector_store(chunks)
            self.save_vector_store()

        self.build_agent()
        self.logger.info(f"{agent_name} agent ready")

    def query(self, question: str) -> dict:
        """
        Query the agent with a question.

        Args:
            question: The question to ask

        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call initialize() first.")

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        final_message = result["messages"][-1]

        source_documents = []
        for msg in result["messages"]:
            if hasattr(msg, "artifact") and msg.artifact:
                if isinstance(msg.artifact, list):
                    source_documents.extend(msg.artifact)
                else:
                    source_documents.append(msg.artifact)

        return {
            "answer": final_message.content,
            "source_documents": source_documents,
        }
