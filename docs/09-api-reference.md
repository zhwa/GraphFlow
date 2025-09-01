# API Reference

This directory contains detailed API documentation for all GraphFlow components.

## Core Components

### StateGraph Class
The main graph builder class that defines workflow structure.

### Node Functions
How to write and configure processing nodes.

### State Management
TypedDict schemas and state handling patterns.

### Commands and Routing
Dynamic flow control and conditional logic.

### Async Support
Asynchronous execution patterns and best practices.

## Organization

The API documentation is organized into these sections:

- **Core Classes** - StateGraph, CompiledGraph, and main interfaces
- **State Management** - TypedDict patterns and state schemas
- **Node Types** - Different node function patterns and signatures
- **Commands** - Command objects for flow control
- **Routing** - Conditional edges and dynamic routing
- **Async** - Asynchronous execution and patterns
- **Utilities** - Helper functions and debugging tools

## Quick Reference

### Basic Graph Creation
```python
from graphflow import StateGraph
from typing import TypedDict

class MyState(TypedDict):
    data: str

graph = StateGraph(MyState)
graph.add_node("process", my_node_function)
graph.set_entry_point("process")
compiled = graph.compile()
```

### Node Function Signatures
```python
# Standard node
def my_node(state: MyState) -> dict:
    return {"data": "processed"}

# Command-returning node
def routing_node(state: MyState) -> Command:
    return Command(update={"data": "updated"}, goto="next")

# Async node
async def async_node(state: MyState) -> dict:
    await some_async_operation()
    return {"data": "async_result"}
```

### Conditional Routing
```python
def route_function(state: MyState) -> str:
    if state["condition"]:
        return "option_a"
    return "option_b"

graph.add_conditional_edges("router", route_function)
```

## Complete API Documentation

For detailed API documentation, see the individual files in this directory once they are created. Each component will have comprehensive documentation including:

- Class and function signatures
- Parameter descriptions
- Return value specifications
- Usage examples
- Common patterns
- Error handling

## Development Status

The API documentation is currently being developed. Priority areas:

1. Core StateGraph API
2. Node function patterns
3. Command system
4. Async patterns
5. Debugging utilities

## Contributing

To contribute API documentation:

1. Follow the established documentation patterns
2. Include comprehensive examples
3. Document all parameters and return values
4. Add error handling information
5. Include performance considerations where relevant
