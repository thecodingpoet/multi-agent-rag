# Multi Agent RAG

A multi-agent retrieval-augmented generation (RAG) system with specialized agents for HR, Finance, and Tech support queries. Includes full observability with Langfuse for debugging and monitoring routing decisions.

## Features

- 🤖 **Specialized Agents**: Separate RAG agents for HR, Finance, and Tech domains
- 🎯 **Orchestrator**: Intelligent routing to the appropriate specialist agent(s)
- 📦 **Vector Stores**: FAISS-based semantic search for each domain
- 📊 **Observability**: Full tracing with Langfuse to debug misrouted questions and track agent performance
- ⭐ **Auto-Evaluation**: Automatic quality scoring (1-10) for every response using LLM-as-a-judge, tracked in Langfuse
