from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from agents.finance_agent import FinanceAgent
from agents.hr_agent import HRAgent
from agents.tech_agent import TechAgent

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Orchestrator:
    """Supervisor agent that coordinates HR, Finance, and Tech specialist agents."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the orchestrator with specialist agents.

        Args:
            llm_model: OpenAI LLM model name for the orchestrator
        """
        self.llm_model = llm_model
        self.orchestrator = None

        # Initialize specialist agents
        print("Initializing specialist agents...")
        self.hr_agent = HRAgent()
        self.hr_agent.initialize()

        self.finance_agent = FinanceAgent()
        self.finance_agent.initialize()

        self.tech_agent = TechAgent()
        self.tech_agent.initialize()

        print("All specialist agents initialized!")

    def build_orchestrator(self):
        """Build the orchestrator agent with wrapped specialist tools."""

        # Wrap each specialist agent as a tool
        @tool
        def handle_hr_query(request: str) -> str:
            """Handle HR-related queries about policies, benefits, vacation, remote work, and performance reviews.

            Use this when the user asks about:
            - Vacation days, PTO, time off
            - Employee benefits, healthcare, retirement
            - Remote work policies
            - Performance reviews
            - Employee onboarding
            - Any HR policies or procedures

            Input: Natural language HR question
            """
            result = self.hr_agent.query(request)
            return result["answer"]

        @tool
        def handle_finance_query(request: str) -> str:
            """Handle Finance-related queries about expenses, reimbursements, purchasing, and payroll.

            Use this when the user asks about:
            - Expense policies and reimbursements
            - Travel expenses
            - Purchasing guidelines and approval processes
            - Payroll information
            - Invoice processing
            - Any finance or accounting procedures

            Input: Natural language finance question
            """
            result = self.finance_agent.query(request)
            return result["answer"]

        @tool
        def handle_tech_query(request: str) -> str:
            """Handle Technical Support queries about IT issues, software, hardware, and access.

            Use this when the user asks about:
            - Password resets and email issues
            - Laptop or hardware problems
            - VPN access and connectivity
            - Software installation
            - IT support contact information
            - Any technical or IT-related issues

            Input: Natural language technical support question
            """
            result = self.tech_agent.query(request)
            return result["answer"]

        # Create orchestrator prompt
        orchestrator_prompt = (
            "You are a helpful company assistant that coordinates specialist agents. "
            "You have access to three specialist teams:\n"
            "1. HR team - handles policies, benefits, vacation, remote work, performance reviews\n"
            "2. Finance team - handles expenses, reimbursements, purchasing, payroll, invoices\n"
            "3. Tech support - handles IT issues, software, hardware, access, passwords\n\n"
            "For each user request:\n"
            "- Identify which specialist(s) can best answer the question\n"
            "- Route to the appropriate specialist agent(s)\n"
            "- If the request spans multiple domains, call multiple specialists\n"
            "- Synthesize the responses into a clear, helpful answer\n\n"
            "Always provide accurate information from the specialist agents."
        )

        model = init_chat_model(self.llm_model, model_provider="openai", temperature=0)

        self.orchestrator = create_agent(
            model,
            tools=[handle_hr_query, handle_finance_query, handle_tech_query],
            system_prompt=orchestrator_prompt,
        )

        print("Orchestrator agent created successfully!")

    def initialize(self):
        """Initialize the complete multi-agent system."""
        print("\n" + "=" * 80)
        print("Initializing Multi-Agent RAG System")
        print("=" * 80)

        self.build_orchestrator()

        print("\n" + "=" * 80)
        print(
            "System ready! You can now ask questions across HR, Finance, and Tech domains."
        )
        print("=" * 80 + "\n")

    def query(self, question: str) -> dict:
        """
        Query the orchestrator with a question.

        Args:
            question: The question to ask

        Returns:
            Dictionary with 'answer' and 'messages'
        """
        if self.orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call initialize() first.")

        result = self.orchestrator.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        final_message = result["messages"][-1]

        return {
            "answer": final_message.content,
            "messages": result["messages"],
        }


if __name__ == "__main__":
    # Initialize the multi-agent system
    orchestrator = Orchestrator()
    orchestrator.initialize()

    # Test queries across different domains
    test_queries = [
        # Single domain queries
        "How many vacation days do I get?",
        "What is the expense reimbursement policy?",
        "How do I reset my email password?",
        # Multi-domain query
        "I need to submit travel expenses for a business trip and my VPN isn't working. Can you help?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        result = orchestrator.query(query)
        print(f"\nAnswer: {result['answer']}")
