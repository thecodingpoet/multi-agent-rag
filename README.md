# Multi Agent RAG

A multi-agent retrieval-augmented generation (RAG) system with specialized agents for HR, Finance, and Tech support queries. Includes full observability with Langfuse for debugging and monitoring routing decisions.

## Features

- 🤖 **Specialized Agents**: Separate RAG agents for HR, Finance, and Tech domains
- 🎯 **Orchestrator**: Intelligent routing to the appropriate specialist agent(s)
- 📦 **Vector Stores**: FAISS-based semantic search for each domain
- 📊 **Observability**: Full tracing with Langfuse to debug misrouted questions and track agent performance
- ⭐ **Auto-Evaluation**: Automatic quality scoring (1-10) for every response using LLM-as-a-judge, tracked in Langfuse

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Langfuse account

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd multi-agent-rag
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:

## Usage

### Interactive Mode

Run the CLI in interactive mode to ask questions:

```bash
uv run python src/multi_agent_system.py
```

Example session:
```
You: What is the vacation policy?
Assistant: According to our HR policy, employees receive...

You: How do I submit an expense report?
Assistant: To submit an expense report, you need to...

You: exit
👋 Goodbye!
```

### Verbose Mode

Enable detailed logging to see agent routing decisions:

```bash
uv run python src/multi_agent_system.py --verbose
```

## Observability with Langfuse

### Viewing Traces

1. Navigate to https://cloud.langfuse.com
2. Select your project
3. Click on "Traces" to see all queries
4. Click on any trace to see:
   - Orchestrator routing decision
   - Which specialist agent(s) were called
   - Document retrieval results
   - LLM calls with prompts and responses
   - Tool invocations
   - Quality scores with reasoning

### Analyzing Quality Scores

Every response receives an automatic quality score (1-10) based on:
- **Accuracy**: Correctness of information
- **Relevance**: How well it addresses the question
- **Completeness**: Whether all aspects are answered
- **Clarity**: Readability and structure

**Special handling**: When the system correctly refuses out-of-scope questions, it receives a high score (8-10) because this is the intended behavior.

To view scores:
1. Open any trace in Langfuse
2. Click the "Scores" tab
3. See `response_quality` score and reasoning

### Debugging Misrouted Questions

If a question is sent to the wrong agent:
1. Find the trace in Langfuse
2. Examine the orchestrator's routing decision
3. Check which tools were called
4. Review the agent selection prompt and response
5. Identify patterns in misrouted queries
6. Adjust orchestrator prompt if needed

## Limitations

- **No Conversation History**: The system processes each query independently without maintaining conversation context. Users cannot ask follow-up questions like "What about for managers?" or "Tell me more", reference previous answers, or build on earlier context within a session.

- **Evaluation Latency**: Every response undergoes automatic quality evaluation, adding 1-3 seconds of latency per query. This is noticeable in interactive sessions as the system waits for GPT-4o-mini to score the response before returning it to the user.

- **Limited Context Retrieval**: The system retrieves only the top 4 most relevant document chunks per query. For complex questions spanning multiple sections of long documents, some relevant information may be missed.

- **No Real-Time Document Updates**: The knowledge base is frozen at startup. If HR updates the vacation policy document or any other source document, the system won't reflect those changes until you manually rebuild the vector store by deleting the existing FAISS index and restarting.

- **Routing Accuracy**: The LLM-based orchestrator can occasionally misroute edge-case questions, particularly cross-domain questions (e.g., "laptop stipend" could route to Finance or Tech), ambiguous terminology that appears in multiple domains, or questions requiring multiple specialists that may only route to one. 


