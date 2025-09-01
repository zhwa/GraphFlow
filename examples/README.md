# GraphFlow Examples

This directory contains examples showcasing GraphFlow parallel execution capabilities.

## Complete Example List

**✅ Updated for GraphFlow Parallel Execution:**

- **[00-quick-start.py](00-quick-start.py)** - **✅ UPDATED** Simple introduction with state reducers
- **[01-test-and-examples.py](01-test-and-examples.py)** - **✅ UPDATED** Comprehensive testing suite with basic functionality validation
- **[02-core-patterns.py](02-core-patterns.py)** - **✅ UPDATED** Essential parallel patterns and techniques
- **[03-hello-world-qa.py](03-hello-world-qa.py)** - **✅ UPDATED** Simple QA system with fallback logic
- **[06-map-reduce.py](06-map-reduce.py)** - **✅ UPDATED** Parallel map-reduce data processing
- **[09-parallel-execution-demo.py](09-parallel-execution-demo.py)** - **✅ UPDATED** Comprehensive parallel showcase

**⚠️ Need GraphFlow Updates:**

- **[04-interactive-chat.py](04-interactive-chat.py)** - Chat agent *(needs TypedDict → dictionary conversion)*
- **[05-research-agent.py](05-research-agent.py)** - Research agent *(needs parallel pattern updates)*
- **[07-async-multi-agent.py](07-async-multi-agent.py)** - Multi-agent *(needs state reducer updates)*
- **[08-llm-integration.py](08-llm-integration.py)** - LLM integration *(needs modernization)*

## Quick Start Guide

### **🚀 RECOMMENDED: Start with Parallel Execution**

```bash
# Best showcase of GraphFlow capabilities
python examples/09-parallel-execution-demo.py
```
**⭐ FLAGSHIP EXAMPLE** - Complete demonstration of:
- Fan-out and fan-in patterns
- State reducers for concurrent updates
- Performance comparison (linear vs parallel)
- Real async execution

### **📚 Learning Path**

```bash
# 1. Basics
python examples/00-quick-start.py        # Simple intro
python examples/03-hello-world-qa.py     # Single node system

# 2. Core Patterns
python examples/02-core-patterns.py      # Essential parallel patterns
python examples/01-test-and-examples.py  # Comprehensive tests

# 3. Real Applications
python examples/06-map-reduce.py         # Practical data processing
```

## Learning Progression

### **🔰 Beginner Path**
1. **00-quick-start.py** - GraphFlow basics with simple state management
2. **03-hello-world-qa.py** - Single node processing with error handling
3. **09-parallel-execution-demo.py** - Core parallel execution concepts

### **🚀 Intermediate Path**
1. **02-core-patterns.py** - Essential parallel patterns (fan-out/fan-in, conditional routing)
2. **01-test-and-examples.py** - Installation validation, basic functionality tests, and comprehensive testing patterns
3. **06-map-reduce.py** - Real-world parallel data processing

### **🎓 Advanced Path**
1. Study the state reducer patterns across all examples
2. Build custom parallel workflows using the demonstrated patterns
3. Optimize performance using parallel execution techniques

## Key GraphFlow Concepts Demonstrated

### **🔄 Parallel Execution**
- **09-parallel-execution-demo.py** - Fan-out/fan-in, performance comparison, async execution
- **06-map-reduce.py** - Parallel data processing with worker pools and result aggregation
- **02-core-patterns.py** - Multiple parallel processing patterns and techniques
- **01-test-and-examples.py** - Parallel execution testing and validation

### **📊 State Management**  
- **00-quick-start.py** - Simple dictionary-based state updates
- **02-core-patterns.py** - State reducers for concurrent list/dict merging
- **06-map-reduce.py** - Complex state aggregation from multiple workers
- **01-test-and-examples.py** - State management testing patterns

### **⚡ Performance & Scalability**
- **09-parallel-execution-demo.py** - Direct linear vs parallel performance comparison
- **06-map-reduce.py** - Scalable data processing with parallel workers
- **02-core-patterns.py** - Efficient parallel pattern implementations

### **🛡️ Error Handling & Reliability**
- **03-hello-world-qa.py** - Graceful fallback when LLM unavailable
- **01-test-and-examples.py** - Comprehensive error handling in parallel contexts
- **02-core-patterns.py** - Robust parallel execution patterns

## ⭐ Showcase Examples

### **Best for Demonstrating GraphFlow:**
1. **09-parallel-execution-demo.py** - Complete parallel execution showcase
2. **06-map-reduce.py** - Real-world parallel data processing
3. **02-core-patterns.py** - Essential parallel patterns library

### **Best for Learning:**
1. **00-quick-start.py** - Gentle introduction to GraphFlow
2. **03-hello-world-qa.py** - Simple single-node processing
3. **01-test-and-examples.py** - Comprehensive testing and validation

## 🚀 GraphFlow Advantages Demonstrated

- **True Parallelism**: Real async execution, not just sequential processing
- **State Reducers**: Declarative concurrent state merging without manual coordination
- **Fan-out/Fan-in**: Natural parallel processing patterns built into the framework
- **Performance**: Measurable speedup over linear execution for parallel workloads
- **Simplicity**: Clean dictionary-based state management without complex schemas

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

### **vs Linear Execution Frameworks**
- ✅ **Parallel Processing** - True async execution vs sequential processing
- ✅ **State Reducers** - Declarative concurrent state management  
- ✅ **Fan-out/Fan-in** - Built-in parallel processing patterns
- ✅ **Better Error Handling** - Structured error management
- ✅ **Cleaner Architecture** - Single responsibility nodes
- ✅ **Testing Support** - Easy to unit test individual components

### **vs LangGraph**
- ✅ **Lighter Weight** - ~500 lines vs complex framework
- ✅ **Zero Dependencies** - Only Python standard library
- ✅ **Better Performance** - Optimized parallel execution core
- ✅ **Simpler API** - Easier to learn and debug

---

**Ready to start? Run `python examples/00-quick-start.py` and begin your GraphFlow journey!** 🚀
