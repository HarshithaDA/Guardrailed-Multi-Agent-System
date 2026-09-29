## Multi-Agent Workflow with Structured Output & Memory

A guardrailed multi-agent workflow built with **LangGraph, CrewAI, PydanticAI, and Mem0**. The system follows a **Planner → Executor → Reviewer** architecture, with LangGraph managing state and execution flow, CrewAI providing the agents, and PydanticAI enforcing strict, type-safe structured outputs.

The workflow includes a hard **maximum of 3 execution attempts** to prevent uncontrolled agent loops, with a safe fallback when the retry budget is exceeded. **Mem0** provides persistent semantic memory using **SQLite for history and Chroma for vector storage**, allowing relevant information from previous executions to be retrieved without passing the entire conversation history to the model.

### Key Features

* Planner → Executor → Reviewer multi-agent architecture
* LangGraph state management and conditional routing
* Maximum execution limit of **N ≤ 3**
* Safe fallback / kill-switch for exceeded execution budget
* Strict PydanticAI schemas for structured output validation
* Reviewer-based approval and retry mechanism
* Persistent Mem0 semantic memory
* SQLite history + Chroma vector storage
* Hugging Face `all-MiniLM-L6-v2` embeddings
* Groq API for LLM inference
* Bounded memory retrieval and concise outputs to control token usage


# Design Decisions

## 1. Multi-Agent Architecture: Planner → Executor → Reviewer

The workflow uses a three-agent architecture consisting of a **Planner, Executor, and Reviewer**.

* **Planner:** Converts the user's request into a short, structured sequence of actionable steps.
* **Executor:** Uses the plan to produce the actual requested deliverable.
* **Reviewer:** Evaluates whether the Executor completed the task and decides whether the result should be approved or retried.

**Design rationale:** Separating planning, execution, and review makes each responsibility explicit and allows the workflow to validate the output before returning it to the user. LangGraph is responsible for orchestration and state transitions, while CrewAI provides the individual agents.

The resulting flow is:

```text
User Task
    ↓
Planner
    ↓
Executor
    ↓
Reviewer
    ↓
Approve ───────→ Final Response
    │
    └─ Retry ──→ Planner
```

---

## 2. LangGraph for Workflow Orchestration

LangGraph was selected to manage the workflow because the system requires explicit state management and conditional routing.

The `GraphState` object stores information shared across the workflow:

```python
class GraphState(TypedDict, total=False):
    user_id: str
    task: str
    execution_count: int
    memory_context: list[str]
    plan: PlannerOutput
    execution: ExecutorOutput
    review: ReviewerOutput
    final_response: str
```

**Design rationale:** Keeping the execution state in one typed graph state makes the workflow predictable and allows the retry mechanism, memory context, planner output, executor output, and reviewer decision to be tracked explicitly.

---

## 3. Execution Loop Cap: N ≤ 3

The workflow uses a hard execution limit:

```python
MAX_EXECUTIONS = 3
```

The execution counter is stored directly in the LangGraph state and incremented before each attempt.

```python
def increment_execution(state):
    current_count = state.get("execution_count", 0)
    return {"execution_count": current_count + 1}
```

The graph then checks whether the maximum execution budget has been exceeded:

```python
def route_after_increment(state):
    if state["execution_count"] > MAX_EXECUTIONS:
        return "fallback"
    return "planner"
```

### Why three executions?

A limit of **three attempts** provides enough opportunity for the Reviewer to request correction when an output is incomplete or invalid, while preventing an uncontrolled agent loop.

For example:

```text
Attempt 1 → Reviewer → retry
Attempt 2 → Reviewer → retry
Attempt 3 → Reviewer → approve → END
```

If another attempt would be required, the workflow activates the fallback instead of continuing indefinitely.

The cap therefore acts as both:

* a **resource-control mechanism**, limiting unnecessary LLM calls and token usage
* a **safety mechanism**, preventing a faulty reviewer/agent interaction from producing an infinite execution loop

The value `3` is also simple to reason about and easy to audit.

---

## 4. Kill-Switch and Safe Fallback

The workflow has a predefined fallback response:

```python
SAFE_FALLBACK = (
    "Execution stopped safely after the maximum retry budget was reached."
)
```

When the execution limit is exceeded, LangGraph routes directly to the fallback node:

```python
def fallback_node(state):
    return {"final_response": SAFE_FALLBACK}
```

The fallback node then terminates the graph.

**Design rationale:** The system should fail safely rather than continue making LLM calls when the workflow cannot produce an acceptable result within its execution budget.

This creates a deterministic termination path:

```text
execution_count > 3
        ↓
    Fallback
        ↓
      END
```

---

## 5. PydanticAI for Structured Output Validation

The agents use PydanticAI to validate their outputs against explicit Pydantic schemas.

A common strict base model is used:

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
```

This prevents unexpected fields and encourages type-safe output.

### Planner schema

```python
class PlannerOutput(StrictModel):
    goal: str = Field(min_length=1)
    steps: list[str] = Field(
        min_length=1,
        max_length=8
    )
```

The Planner therefore has a clearly defined contract:

```text
goal → string
steps → list of actionable strings
```

### Executor schema

```python
class ExecutorOutput(StrictModel):
    result: str = Field(
        min_length=20,
        max_length=4000
    )
    completed_steps: list[str] = Field(
        min_length=1,
        max_length=8
    )
```

The Executor must provide both:

1. the actual requested result
2. the steps that were completed

The `max_length=4000` constraint was intentionally introduced to prevent very large structured responses from causing excessive token usage or JSON/tool-call truncation.

### Reviewer schema

```python
class ReviewerOutput(StrictModel):
    decision: Literal["approve", "retry"]
    feedback: str = Field(min_length=1)
```

Using:

```python
Literal["approve", "retry"]
```

makes the review decision deterministic rather than allowing arbitrary natural-language decisions.

---

## 6. Why Strict Schemas Were Chosen

LLMs naturally produce flexible text, but a multi-agent system requires predictable interfaces between agents.

Without validation, the Planner could return:

```text
Here is what I think you should do...
```

instead of a structured plan.

Similarly, the Reviewer could return:

```text
The response looks mostly good, but perhaps...
```

which would make routing difficult.

With PydanticAI, each component has a defined contract.

```text
Planner
   ↓
PlannerOutput
   ↓
Executor
   ↓
ExecutorOutput
   ↓
Reviewer
   ↓
ReviewerOutput
```

This makes the workflow easier to validate, debug, and extend.

---

## 7. Reviewer-Based Retry Design

The Reviewer does not automatically retry every response. It only returns one of two decisions:

```text
approve
retry
```

The routing logic is:

```python
def route_after_review(state):
    if state["review"].decision == "approve":
        return "complete"
    return "retry"
```

**Design rationale:** The Reviewer provides a quality-control layer between execution and completion. A retry occurs only when the output is considered incomplete or unusable.

This prevents unnecessary retries when the Executor has already produced an acceptable response.

---

## 8. Persistent Memory with Mem0

Mem0 was selected to provide persistent semantic memory across workflow executions.

The implementation uses:

* **Mem0** for memory management
* **SQLite** for persistent history
* **Chroma** for vector storage
* **Hugging Face `all-MiniLM-L6-v2`** for embeddings

The storage configuration includes:

```python
MEMORY_DB = MEMORY_DIR / "history.sqlite3"
CHROMA_DIR = MEMORY_DIR / "chroma"
```

Memory is retrieved before the Planner runs:

```python
memory_context = get_memory_context(
    task,
    user_id
)
```

The retrieved context is then provided to the Planner.

After a successful review, the completed result is saved:

```python
save_memory(
    task=state["task"],
    result=state["execution"].result,
    user_id=state["user_id"],
)
```

**Design rationale:** The system can reuse relevant information from previous executions without requiring the entire previous conversation to be passed to the LLM every time.

---

## 9. Limiting Memory Context

The memory retrieval function limits the amount of retrieved context:

```python
return context[:5]
```

This was intentionally done to prevent memory retrieval from continuously increasing the prompt size.

The design therefore combines:

```text
Persistent Memory
       +
Semantic Retrieval
       +
Bounded Context
```

rather than passing the entire memory database into every LLM request.

This helps control token consumption as the number of stored memories increases.

---

## 10. `infer=False` for Memory Storage

Mem0 is configured to store the completed result directly:

```python
memory.add(
    [{"role": "user", "content": memory_text}],
    user_id=user_id,
    infer=False,
)
```

`infer=False` was chosen because the initial Mem0 configuration attempted additional LLM-based memory extraction and encountered Groq token/TPM limitations.

By storing the already-structured successful result directly, the system avoids an unnecessary additional LLM call during memory storage.

This provides a simpler and more predictable memory pipeline:

```text
Approved Result
      ↓
Direct Mem0 Storage
      ↓
SQLite + Chroma
```

---

## 11. Concise Agent Prompts

The Planner, Executor, and Reviewer prompts were deliberately kept concise.

This decision was made after observing that excessively long generated outputs could cause structured-output failures and increase token consumption.

For example, the Executor is explicitly instructed to:

```text
Produce the actual requested deliverable.
Keep the result concise and usable.
Do not merely describe the plan.
Do not return only a summary.
```

The Pydantic schema also limits the size of the Executor result.

**Design rationale:** The system prioritizes useful, bounded outputs over unnecessarily long responses.

---

## 12. Groq as the LLM Provider

Groq was used as the LLM provider rather than requiring a local Ollama installation.

The implementation uses OpenAI-compatible API endpoints through Groq.

This allows the workflow to run directly in Google Colab without requiring a locally installed model server.

The architecture remains provider-oriented:

```text
CrewAI / PydanticAI
        ↓
OpenAI-compatible interface
        ↓
Groq
```

This also keeps the agent framework separate from the underlying model provider.

---

## 13. Separation of Responsibilities

Each framework has a specific responsibility rather than using all frameworks for the same purpose:

| Technology       | Responsibility                                         |
| ---------------- | ------------------------------------------------------ |
| **LangGraph**    | Workflow graph, state, conditional routing, retry loop |
| **CrewAI**       | Planner, Executor, and Reviewer agents                 |
| **PydanticAI**   | Structured output validation                           |
| **Pydantic**     | Strict data schemas and type constraints               |
| **Mem0**         | Persistent semantic memory                             |
| **SQLite**       | Persistent memory/history storage                      |
| **Chroma**       | Vector storage and semantic retrieval                  |
| **Hugging Face** | Text embeddings                                        |
| **Groq**         | LLM inference                                          |

This separation makes the system modular and easier to replace or extend.

---

## 14. Error and Failure 

The workflow follows a **fail-safe rather than fail-open** design.

Potential failure paths are bounded:

```text
Invalid/incomplete output
        ↓
Reviewer
        ↓
Retry
        ↓
Maximum 3 attempts
        ↓
Fallback
        ↓
END
```

Structured validation prevents malformed outputs from silently propagating between agents, while the execution budget prevents repeated failures from becoming an uncontrolled loop.

---

## 15. Overall Design Goal

The overall design balances three requirements:

### Reliability

Strict schemas and reviewer validation ensure that agents communicate through predictable interfaces.

### Safety

The execution budget and fallback mechanism prevent uncontrolled agent loops and excessive resource consumption.

### Persistence

Mem0 allows successful results to be reused across executions without repeatedly sending the entire historical context to the LLM.

### Architecture 

```text
                    ┌──────────────┐
                    │     Mem0     │
                    │ SQLite+Chroma│
                    └──────┬───────┘
                           │
                           ▼
User Task ──────────► LangGraph State
                           │
                           ▼
                      ┌─────────┐
                      │ Planner │
                      │ CrewAI  │
                      └────┬────┘
                           │
                    PydanticAI
                    validation
                           │
                           ▼
                      ┌─────────┐
                      │Executor │
                      │ CrewAI  │
                      └────┬────┘
                           │
                    PydanticAI
                    validation
                           │
                           ▼
                      ┌─────────┐
                      │Reviewer │
                      │ CrewAI  │
                      └────┬────┘
                           │
                  ┌────────┴────────┐
                  │                 │
               APPROVE            RETRY
                  │                 │
                  ▼                 ▼
             Save to Mem0      Counter + 1
                  │                 │
                  ▼                 │
            Final Response          │
                                    │
                    ┌───────────────┘
                    │
                    ▼
              Counter > 3
                    │
                    ▼
              Safe Fallback
                    │
                    ▼
                   END
```



