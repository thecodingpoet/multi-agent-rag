"""Response quality evaluator using LLM-as-a-judge."""

import json
import logging

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langfuse import get_client

load_dotenv()

NEUTRAL_ERROR_SCORE = 5


class ResponseEvaluator:
    """Evaluates RAG response quality using GPT-4o-mini as judge."""

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        """
        Initialize the response evaluator.

        Args:
            judge_model: OpenAI model to use as judge (default: gpt-4o-mini)
        """
        self.judge_model = judge_model
        self.logger = logging.getLogger("agents.evaluator")
        self.langfuse = get_client()

        self.llm = init_chat_model(
            self.judge_model,
            model_provider="openai",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        self.logger.info(f"Response evaluator initialized with {judge_model}")

    def create_evaluation_prompt(self, question: str, answer: str) -> str:
        """
        Create the evaluation prompt for the judge LLM.

        Args:
            question: Original user question
            answer: RAG system's answer

        Returns:
            Formatted evaluation prompt
        """
        return (
            "You are evaluating a RAG (Retrieval-Augmented Generation) system's response quality. "
            "Your task is to rate the answer on a scale of 1-10 based on these criteria: "
            "Accuracy (is the information correct and factual?), "
            "Relevance (does it directly address the question?), "
            "Completeness (does it fully answer the question?), "
            "and Clarity (is it easy to understand and well-structured?). "
            f"Question: {question} "
            f"Answer: {answer} "
            "IMPORTANT: If the answer correctly declines to answer because the question is outside the system's scope "
            "(not related to HR, Finance, or Tech support), this should be scored HIGH (8-10) because the system is "
            "working correctly by refusing out-of-scope questions. Only score low if the answer is wrong, unclear, or "
            "incorrectly refuses to answer an in-scope question. "
            "Provide your evaluation in JSON format with 'score' (number from 1-10) and 'reasoning' (brief explanation) fields. "
            "Be strict but fair. A score of 10 should be reserved for exceptional answers."
        )

    def evaluate(self, question: str, answer: str) -> dict:
        """
        Evaluate a RAG response and return score with reasoning.

        Args:
            question: Original user question
            answer: RAG system's answer

        Returns:
            Dictionary with 'score' (1-10) and 'reasoning'
        """
        try:
            prompt = self.create_evaluation_prompt(question, answer)

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            result = json.loads(content)

            self.logger.debug(
                f"Evaluation: score={result['score']}, reasoning={result['reasoning']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {
                "score": NEUTRAL_ERROR_SCORE,
                "reasoning": f"Evaluation error: {str(e)}",
            }

    def save_to_langfuse(self, trace_id: str, score: int, reasoning: str):
        """
        Save evaluation score to Langfuse.

        Args:
            trace_id: Langfuse trace ID to attach score to
            score: Quality score (1-10)
            reasoning: Reasoning for the score
        """
        try:
            self.langfuse.create_score(
                trace_id=trace_id,
                name="response_quality",
                value=score,
                comment=reasoning,
            )

            self.logger.info(
                f"Score saved to Langfuse: trace_id={trace_id}, score={score}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save score to Langfuse: {e}")
