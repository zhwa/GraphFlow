# GraphFlow Examples

This directory contains all GraphFlow examples in a simple, flat structure with unified numbering.

## Complete Example List

All examples are in the same directory with clear numbering:

- **[00-quick-start.py](00-quick-start.py)** - Gentle introduction with simple conversational agent
- **[01-test-and-examples.py](01-test-and-examples.py)** - Comprehensive tests + 5 core patterns
- **[02-core-patterns.py](02-core-patterns.py)** - Foundation patterns and advanced techniques
- **[03-hello-world-qa.py](03-hello-world-qa.py)** - Simple question-answering system *(ported)*
- **[04-interactive-chat.py](04-interactive-chat.py)** - Conversational agent with memory *(ported)*
- **[05-research-agent.py](05-research-agent.py)** - Intelligent agent with web search *(ported)*
- **[06-map-reduce.py](06-map-reduce.py)** - Parallel processing and data aggregation *(ported)*
- **[07-async-multi-agent.py](07-async-multi-agent.py)** - Concurrent multi-agent coordination *(ported)*
- **[08-llm-integration.py](08-llm-integration.py)** - **NEW!** LLM integration (OpenAI, Anthropic, Ollama) 🤖

*(ported) = Adapted from PocketFlow cookbook with GraphFlow improvements*

## Quick Start Guide

### **Complete Beginner?**
```bash
python examples/00-quick-start.py
```
A gentle 5-minute introduction that shows you GraphFlow basics.

### **Want to Test Installation?**
```bash
python examples/01-test-and-examples.py
```
Validates your setup and demonstrates 5 core patterns:
- Basic counter operations
- Conditional routing  
- Chat agents with memory
- Multi-step workflows
- Error handling

### **Ready for Real Examples?**
```bash
# Foundation patterns
python examples/02-core-patterns.py

# Simple applications
python examples/03-hello-world-qa.py
python examples/04-interactive-chat.py

# Advanced applications
python examples/05-research-agent.py
python examples/06-map-reduce.py
python examples/07-async-multi-agent.py
```

## Learning Progression

### **🔰 Beginner Path (Start Here)**
1. **00-quick-start.py** - Learn the basics in 5 minutes
2. **01-test-and-examples.py** - See 5 core patterns and validate installation
3. **03-hello-world-qa.py** - Simple single-node example
4. **04-interactive-chat.py** - Understand state flow and conversation loops

### **🚀 Intermediate Path**
1. **02-core-patterns.py** - Comprehensive foundation patterns
2. **05-research-agent.py** - Conditional routing and decision making
3. **06-map-reduce.py** - Parallel processing and aggregation patterns

### **🎓 Advanced Path**
1. **07-async-multi-agent.py** - Concurrent execution and agent coordination
2. Combine patterns to build custom applications
3. Extend with your own nodes and workflows

## Example Categories

### **State Management**
- **00-quick-start.py** - Basic state updates
- **04-interactive-chat.py** - Conversation state and memory
- **01-test-and-examples.py** - Counter and workflow state patterns

### **Flow Control**
- **02-core-patterns.py** - Sequential and conditional flows
- **05-research-agent.py** - Dynamic routing based on confidence
- **01-test-and-examples.py** - Error handling and recovery flows

### **Agent Patterns**
- **04-interactive-chat.py** - Simple conversational agent
- **05-research-agent.py** - Research and analysis agent
- **07-async-multi-agent.py** - Multi-agent coordination

### **Data Processing**
- **03-hello-world-qa.py** - Simple text processing
- **06-map-reduce.py** - Parallel data processing and aggregation
- **01-test-and-examples.py** - Multi-step data transformation

### **Async & Concurrency**
- **07-async-multi-agent.py** - Queue-based communication and coordination
- **06-map-reduce.py** - Parallel processing patterns

## Key Concepts Demonstrated

### **Type Safety**
All examples use TypedDict for state schemas:
```python
class MyState(TypedDict):
    input: str
    result: str
    count: int
```

### **Immutable State Updates**
Examples show proper state updating:
```python
def my_node(state: MyState) -> dict:
    return {
        "result": f"Processed: {state['input']}",
        "count": state["count"] + 1
    }
```

### **Command Objects**
Advanced examples use Commands for flow control:
```python
def my_node(state: MyState) -> Command:
    if state["count"] > 5:
        return Command(update={...}, goto=END)
    else:
        return Command(update={...}, goto="continue")
```

### **Conditional Routing**
Dynamic flow control based on state:
```python
def route_condition(state: MyState) -> str:
    return "success" if state["result"] else "retry"

graph.add_conditional_edges("node", route_condition, {
    "success": "finish",
    "retry": "process_again"
})
```

## Running Examples

### **Prerequisites**
- Python 3.8+
- GraphFlow framework (graphflow.py in root directory)

### **Basic Usage**
```bash
# From project root
python examples/00-quick-start.py

# Or with explicit path
cd examples
python 00-quick-start.py
```

### **Expected Behavior**
All examples now work correctly with relative imports and will run successfully:
```bash
# Should work perfectly
python examples/00-quick-start.py
python examples/01-test-and-examples.py
python examples/03-hello-world-qa.py
# ... etc
```

The examples automatically import GraphFlow from the parent directory.

## Creating New Examples

### **Naming Convention**
Follow the unified numbering scheme:
- `00-09` - Getting started and basic tutorials
- `10-19` - Intermediate patterns  
- `20-29` - Advanced patterns
- `30+` - Specialized use cases

### **Example Template**
```python
"""
GraphFlow Example: [Brief Description]

This example demonstrates:
- Key concept 1
- Key concept 2
"""

from graphflow import StateGraph, Command, END
from typing import TypedDict

class ExampleState(TypedDict):
    input_data: str
    result: str

def example_node(state: ExampleState) -> dict:
    """Brief description of what this node does."""
    return {"result": f"Processed: {state['input_data']}"}

def build_graph():
    """Build and configure the example graph."""
    graph = StateGraph(ExampleState)
    graph.add_node("example", example_node)
    graph.set_entry_point("example")
    return graph.compile()

def main():
    """Main function with test cases."""
    app = build_graph()
    result = app.invoke({"input_data": "test", "result": ""})
    print(result)

if __name__ == "__main__":
    main()
```

## Getting Help

### **Documentation**
- **[Main Documentation](../docs/)** - Complete guides and reference
- **[Quick Start Guide](../docs/01-quick-start.md)** - Step-by-step tutorial
- **[Architecture Overview](../docs/04-architecture.md)** - How GraphFlow works

### **Next Steps**
After exploring these examples:
1. Read the [documentation](../docs/) for deeper understanding
2. Try modifying existing examples
3. Build your own GraphFlow applications
4. Contribute new examples back to the project

## GraphFlow Advantages Highlighted

These examples showcase GraphFlow's key benefits:

### **vs Original PocketFlow Examples**
- ✅ **Type Safety** - Strong TypedDict schemas vs dynamic dictionaries
- ✅ **Immutable State** - Clear state flow without mutation  
- ✅ **Better Error Handling** - Structured error management
- ✅ **Cleaner Architecture** - Single responsibility nodes
- ✅ **Testing Support** - Easy to unit test individual components

### **vs LangGraph**
- ✅ **Lighter Weight** - ~500 lines vs complex framework
- ✅ **Zero Dependencies** - Only Python standard library
- ✅ **Better Performance** - Embedded PocketFlow core
- ✅ **Simpler API** - Easier to learn and debug

---

**Ready to start? Run `python examples/00-quick-start.py` and begin your GraphFlow journey!** 🚀
