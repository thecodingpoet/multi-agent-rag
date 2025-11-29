#!/usr/bin/env python3
"""
Multi-Agent RAG System CLI

A command-line interface for querying the multi-agent RAG system.
Allows users to ask questions across HR, Finance, and Tech domains.
"""

import argparse
import logging
import sys

from agents.orchestrator import Orchestrator
from utils.logger import setup_logger


def print_banner():
    """Print welcome banner."""
    print("\nWelcome! I can help you with:")
    print("  • HR: Vacation, benefits, remote work, performance reviews")
    print("  • Finance: Expenses, reimbursements, purchasing, payroll")
    print("  • Tech: IT support, passwords, VPN, software installation")
    print("\nType 'exit' or 'quit' to end the session.")
    print("-" * 80 + "\n")


def setup_logging(verbose: bool) -> logging.Logger:
    """
    Configure logging for all components.

    Args:
        verbose: Enable verbose logging if True

    Returns:
        Logger instance for CLI
    """
    log_level = logging.DEBUG if verbose else logging.WARNING

    for component in ["hr", "finance", "tech", "orchestrator", "evaluator"]:
        setup_logger(f"agents.{component}", level=log_level)

    return setup_logger("cli", level=log_level)


def handle_query(
    orchestrator: Orchestrator, query: str, logger: logging.Logger
) -> bool:
    """
    Process a single query.

    Args:
        orchestrator: The orchestrator instance
        query: User query string
        logger: Logger instance

    Returns:
        True to continue, False to exit
    """
    query = query.strip()

    if query.lower() in ["exit", "quit", "q"]:
        print("\n👋 Goodbye!\n")
        return False

    if not query:
        return True

    try:
        result = orchestrator.query(query)
        print(f"Assistant: {result['answer']}\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        logger.error(f"Error processing query: {str(e)}")

    return True


def run_interactive_loop(orchestrator: Orchestrator, logger: logging.Logger):
    """
    Run the main interactive query loop.

    Args:
        orchestrator: The orchestrator instance
        logger: Logger instance
    """
    while True:
        try:
            query = input("\nYou: ")
            if not handle_query(orchestrator, query, logger):
                break
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break


def main():
    """Run the CLI application."""
    parser = argparse.ArgumentParser(description="Multi-Agent RAG System CLI")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        orchestrator = Orchestrator()
        orchestrator.initialize()

        print_banner()
        run_interactive_loop(orchestrator, logger)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}\n")
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
