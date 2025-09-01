# GraphFlow Documentation

## Overview

**GraphFlow** is a LangGraph-like agent framework built on PocketFlow's minimalist foundation. It provides state-based graph execution with conditional routing, dynamic flow control, and async support while maintaining simplicity and avoiding heavy dependencies like LangChain.

## Key Features

### 1. **State Management**
- Type-safe state schemas using TypedDict
- Automatic state merging and updates
- List concatenation by default for array fields

### 2. **Node Types**
- **GraphNode**: Standard synchronous nodes
- **AsyncGraphNode**: Asynchronous nodes with async/await support
- Built-in retry mechanisms and fallback handling (from PocketFlow)

### 3. **Routing & Control Flow**
- **Direct Edges**: Simple node-to-node transitions
- **Conditional Edges**: Dynamic routing based on state
- **Command Objects**: Combine state updates with routing decisions
- **Send Objects**: For map-reduce workflows (planned feature)

### 4. **Graph Construction**
- Fluent API for building graphs
- Compile-time validation
- Visual graph representation support (planned)

## Core Classes

### StateGraph[StateT]

The main graph builder class.

```python
from graphflow import StateGraph
from typing import TypedDict, List

class MyState(TypedDict):
    messages: List[str]
    counter: int

graph = StateGraph(MyState)
```

**Methods:**
- `add_node(name, func, **kwargs)`: Add a node function
- `add_edge(from_node, to_node)`: Add direct edge
- `add_conditional_edges(from_node, condition, path_map=None)`: Add conditional routing
- `set_entry_point(node_name)`: Set starting node
- `compile()`: Create executable graph

### GraphNode

Enhanced node with state management capabilities.

```python
def my_node(state: MyState) -> dict:
    return {
        "counter": state["counter"] + 1,
        "messages": state["messages"] + ["Processed"]
    }

graph.add_node("process", my_node)
```

**Return Types:**
- `dict`: Update state with these values
- `Command`: Combine state update with routing
- `str`: Direct routing to named node
- `None`: Continue with default routing

### Command

Combines state updates with routing decisions.

```python
from graphflow import Command, END

def decision_node(state: MyState) -> Command:
    if state["counter"] >= 5:
        return Command(
            update={"messages": state["messages"] + ["Done!"]},
            goto=END
        )
    else:
        return Command(
            update={"counter": state["counter"] + 1},
            goto="continue_processing"
        )
```

**Fields:**
- `update`: Dictionary of state updates
- `goto`: Next node name, list of nodes, or END
- `resume`: Resume values for interrupts (planned)

### ConditionalEdge

Routes based on state evaluation.

```python
def route_by_priority(state: MyState) -> str:
    if state["counter"] > 10:
        return "high_priority"
    elif state["counter"] > 5:
        return "medium_priority"
    else:
        return "low_priority"

graph.add_conditional_edges("analyzer", route_by_priority)

# With path mapping
graph.add_conditional_edges(
    "analyzer", 
    lambda state: state["priority_level"],
    {"urgent": "urgent_handler", "normal": "normal_handler"}
)
```

## Usage Patterns

### 1. Simple Sequential Processing

```python
from graphflow import StateGraph
from typing import TypedDict, List

class State(TypedDict):
    text: str
    processed: bool

def process_text(state: State) -> dict:
    return {
        "text": state["text"].upper(),
        "processed": True
    }

graph = StateGraph(State)
graph.add_node("process", process_text)
graph.set_entry_point("process")

result = graph.compile().invoke({
    "text": "hello world",
    "processed": False
})
```

### 2. Conditional Branching

```python
def analyze(state: State) -> dict:
    return {"analysis_done": True}

def route_analysis(state: State) -> str:
    if "urgent" in state["text"].lower():
        return "urgent_handler"
    else:
        return "normal_handler"

def urgent_handler(state: State) -> dict:
    return {"text": f"URGENT: {state['text']}"}

def normal_handler(state: State) -> dict:
    return {"text": f"Normal: {state['text']}"}

graph = StateGraph(State)
graph.add_node("analyze", analyze)
graph.add_node("urgent_handler", urgent_handler)
graph.add_node("normal_handler", normal_handler)

graph.add_conditional_edges("analyze", route_analysis)
graph.set_entry_point("analyze")
```

### 3. Loops and Iterations

```python
def counter_node(state: State) -> Command:
    new_count = state["counter"] + 1
    
    if new_count < 5:
        return Command(
            update={"counter": new_count},
            goto="counter_node"  # Loop back
        )
    else:
        return Command(
            update={"counter": new_count, "done": True},
            goto=END
        )

graph = StateGraph(State)
graph.add_node("counter_node", counter_node)
graph.set_entry_point("counter_node")
```

### 4. Agent-like Workflows

```python
class AgentState(TypedDict):
    query: str
    thoughts: List[str]
    actions: List[str]
    final_answer: str

def think(state: AgentState) -> dict:
    thought = f"Analyzing query: {state['query']}"
    action = "search" if "?" in state["query"] else "respond"
    
    return {
        "thoughts": state["thoughts"] + [thought],
        "actions": state["actions"] + [action]
    }

def should_continue(state: AgentState) -> str:
    last_action = state["actions"][-1] if state["actions"] else ""
    
    if last_action == "respond":
        return "respond"
    else:
        return "search"

def search(state: AgentState) -> dict:
    return {
        "thoughts": state["thoughts"] + ["Searching for information..."]
    }

def respond(state: AgentState) -> dict:
    return {
        "final_answer": "Here's my response based on the analysis."
    }

graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_node("search", search)
graph.add_node("respond", respond)

graph.add_conditional_edges("think", should_continue)
graph.add_edge("search", "think")  # Loop back to think after searching
graph.set_entry_point("think")
```

## Async Support

GraphFlow supports async nodes for I/O-bound operations:

```python
import asyncio
from graphflow import StateGraph

async def async_api_call(state: State) -> dict:
    # Simulate API call
    await asyncio.sleep(0.1)
    return {"api_result": "success"}

graph = StateGraph(State)
graph.add_node("api_call", async_api_call)
graph.set_entry_point("api_call")

# Use ainvoke for async execution
result = await graph.compile().ainvoke(initial_state)
```

## Error Handling and Retries

Built-in retry mechanisms from PocketFlow:

```python
def unreliable_node(state: State) -> dict:
    # This might fail sometimes
    if random.random() < 0.5:
        raise Exception("Temporary failure")
    return {"result": "success"}

# Add node with retry configuration
graph.add_node("unreliable", unreliable_node, max_retries=3, wait=1)
```

## Best Practices

### 1. State Design
- Keep state flat when possible
- Use TypedDict for type safety
- Avoid deeply nested structures
- Use lists for accumulating results

### 2. Node Functions
- Return state updates as dictionaries
- Use Command for complex routing logic
- Keep nodes focused on single responsibilities
- Handle edge cases gracefully

### 3. Graph Structure
- Start simple, add complexity incrementally
- Use meaningful node names
- Add conditional edges for decision points
- Test individual nodes before building full graphs

### 4. Error Handling
- Let PocketFlow's retry mechanism handle transient errors
- Use fallback nodes for permanent failures
- Log important state transitions
- Validate inputs at graph boundaries

## Comparison with LangGraph

| Feature | GraphFlow | LangGraph |
|---------|-----------|-----------|
| **Dependencies** | PocketFlow only | LangChain + extras |
| **Complexity** | Minimal, focused | Feature-rich, complex |
| **State Management** | Simple dict-based | Advanced with reducers |
| **Conditional Routing** | ✅ | ✅ |
| **Command Objects** | ✅ | ✅ |
| **Send/Map-Reduce** | Planned | ✅ |
| **Streaming** | Basic | Advanced |
| **Persistence** | Not built-in | ✅ |
| **Human-in-loop** | Not built-in | ✅ |
| **Learning Curve** | Low | Medium-High |

## Future Enhancements

1. **Enhanced Send Support**: Full map-reduce workflows
2. **Graph Visualization**: Mermaid diagram generation
3. **Streaming Support**: Real-time state updates
4. **Persistence Layer**: State checkpointing
5. **Human-in-the-Loop**: Interactive workflows
6. **Performance Optimizations**: Parallel execution
7. **Development Tools**: Debugging and profiling

## Examples

See `examples.py` for comprehensive examples including:
- Simple chat agents
- Conditional routing
- Command-based control flow
- Agent-like workflows
- Error handling patterns

## Testing

Run the test suite:

```bash
python test_graphflow.py
```

Run examples:

```bash
python examples.py
```

## Contributing

GraphFlow follows PocketFlow's philosophy of simplicity and focus. When contributing:

1. Keep the API minimal and intuitive
2. Maintain compatibility with PocketFlow patterns
3. Add comprehensive tests for new features
4. Document all public APIs
5. Focus on common use cases over edge cases

## License

Same as PocketFlow (check PocketFlow/LICENSE)
