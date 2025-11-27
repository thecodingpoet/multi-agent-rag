#!/usr/bin/env python3
"""
Multi-Agent RAG System CLI

A command-line interface for querying the multi-agent RAG system.
Allows users to ask questions across HR, Finance, and Tech domains.
"""

import sys
from pathlib import Path

from agents.orchestrator import Orchestrator


def print_banner():
    """Print welcome banner."""
    print("Welcome! I can help you with:")
    print("  • HR: Vacation, benefits, remote work, performance reviews")
    print("  • Finance: Expenses, reimbursements, purchasing, payroll")
    print("  • Tech: IT support, passwords, VPN, software installation")
    print("\nType 'exit' or 'quit' to end the session.")
    print("-" * 80 + "\n")


def main():
    """Run the CLI application."""
    try:
        # Print banner
        print_banner()

        # Initialize the orchestrator
        print("Initializing the multi-agent system...")
        orchestrator = Orchestrator()
        orchestrator.initialize()

        # Main interaction loop
        while True:
            try:
                # Get user input
                query = input("\n💬 Your question: ").strip()

                # Check for exit commands
                if query.lower() in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break

                # Skip empty queries
                if not query:
                    continue

                # Process query
                print("\n🤔 Processing your question...\n")
                result = orchestrator.query(query)

                # Display answer
                print("=" * 80)
                print("✅ Answer:")
                print("=" * 80)
                print(f"{result['answer']}")
                print("=" * 80)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {str(e)}\n")
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
