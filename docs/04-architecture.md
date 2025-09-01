# GraphFlow Architecture Overview

This document explains how GraphFlow works under the hood, its design principles, and the relationship between its components.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GraphFlow API                        │
│  StateGraph, Command, ConditionalEdge, Send            │
├─────────────────────────────────────────────────────────┤
│                Enhanced Node Layer                      │
│         GraphNode, AsyncGraphNode                       │
├─────────────────────────────────────────────────────────┤
│                PocketFlow Core (Embedded)               │
│        BaseNode, Node, Flow, AsyncNode                  │
└─────────────────────────────────────────────────────────┘
```

## Design Philosophy

GraphFlow embeds and extends PocketFlow's minimalist philosophy:

### 1. **Simplicity First**
- Single file (~500 lines total)
- No external dependencies
- Clear, readable code
- Minimal API surface

### 2. **State-Centric Design**
- All data flows through shared state
- Nodes read and write to state
- Predictable data flow
- Easy to debug and understand

### 3. **Function-Based Nodes**
- Nodes are pure Python functions
- No complex class hierarchies
- Easy to test in isolation
- Straightforward to compose

### 4. **Composable Architecture**
- Build complex workflows from simple parts
- Reusable node functions
- Flexible routing strategies
- Easy to extend and modify

## Core Components

### StateGraph

The main orchestrator that builds and manages graphs.

```python
class StateGraph(Generic[StateT]):
    def __init__(self, state_schema: type = None):
        self.state_schema = state_schema
        self.nodes: Dict[str, Union[GraphNode, AsyncGraphNode]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionalEdge] = {}
        self.entry_point: Optional[str] = None
        # ... other internal state
```

**Responsibilities:**
- Graph construction via fluent API
- Node and edge management
- Validation and compilation
- Creation of executable graphs

### GraphNode

Enhanced nodes that understand state management.

```python
class GraphNode(Node):  # Inherits from PocketFlow's Node
    def __init__(self, func: Callable[[StateT], Any], name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.name = name
    
    def prep(self, shared: StateT) -> StateT:
        """Pass entire state to node function."""
        return shared
    
    def exec(self, state: StateT) -> Any:
        """Execute the node function."""
        return self.func(state)
    
    def post(self, shared: StateT, prep_res: StateT, exec_res: Any) -> str:
        """Update state and determine next action."""
        # Handle different return types: dict, Command, str
        # Update shared state appropriately
        # Return routing decision
```

**Key Features:**
- Wraps user functions in PocketFlow's execution model
- Handles different return types (dict, Command, str)
- Manages state updates automatically
- Provides retry and error handling

### Command Objects

Enable combining state updates with routing decisions.

```python
@dataclass
class Command:
    update: Optional[Dict[str, Any]] = None
    goto: Optional[Union[str, List[str], Send, List[Send]]] = None
    resume: Optional[Dict[str, Any]] = None
```

**Usage Pattern:**
```python
def decision_node(state: MyState) -> Command:
    if should_continue(state):
        return Command(
            update={"step": state["step"] + 1},
            goto="next_step"
        )
    else:
        return Command(
            update={"result": "completed"},
            goto=END
        )
```

### ConditionalEdge

Handles dynamic routing based on state.

```python
class ConditionalEdge:
    def __init__(self, condition: Callable[[StateT], Union[str, List[str]]], 
                 path_map: Optional[Dict[Any, str]] = None):
        self.condition = condition
        self.path_map = path_map or {}
    
    def __call__(self, state: StateT) -> Union[str, List[str]]:
        result = self.condition(state)
        
        if self.path_map and result in self.path_map:
            return self.path_map[result]
        
        return result
```

## Execution Flow

### 1. Graph Construction

```python
# User builds graph
graph = StateGraph(MyState)
graph.add_node("step1", step1_func)
graph.add_node("step2", step2_func)
graph.add_conditional_edges("step1", route_func)
graph.set_entry_point("step1")

# Compilation creates executable version
compiled = graph.compile()
```

**What happens internally:**
1. StateGraph stores node functions and routing rules
2. Compilation wraps functions in GraphNode instances
3. Creates StateFlow for execution orchestration
4. Validates graph structure

### 2. Graph Execution

```python
result = compiled.invoke(initial_state)
```

**Execution steps:**
1. **Start**: Begin at entry point with initial state
2. **Node Execution**: 
   - Call node function with current state
   - Process return value (dict/Command/str)
   - Update shared state
3. **Routing Decision**:
   - Check for Command.goto
   - Evaluate conditional edges
   - Follow direct edges
4. **Next Node**: Move to determined next node
5. **Repeat**: Continue until END or no next node

### 3. State Management

State flows through the graph and gets updated at each step:

```python
# Initial state
state = {"messages": [], "counter": 0}

# Node 1 updates
state = {"messages": ["Hello"], "counter": 1}

# Node 2 updates (list concatenation)
state = {"messages": ["Hello", "World"], "counter": 2}
```

**Update Rules:**
- **Dictionaries**: Merged (later values override)
- **Lists**: Concatenated by default
- **Other types**: Replaced

## PocketFlow Integration

GraphFlow embeds PocketFlow's core classes directly:

### Embedded Classes
- `BaseNode`: Core node abstraction
- `Node`: Node with retry/fallback
- `Flow`: Basic flow orchestration
- `AsyncNode`: Async execution support

### Why Embedded?
1. **Zero Dependencies**: Self-contained single file
2. **Customization**: Can modify for GraphFlow needs
3. **Simplicity**: No import complexity
4. **Performance**: Direct function calls

### Extension Strategy
GraphFlow extends PocketFlow by:
- Adding state-aware node wrappers
- Implementing conditional routing
- Adding Command/Send abstractions
- Providing LangGraph-like API

## Async Support

GraphFlow supports async execution throughout:

```python
class AsyncGraphNode(AsyncNode, GraphNode):
    async def exec_async(self, state: StateT) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(state)
        else:
            return self.func(state)
```

**Async Features:**
- Async node functions
- Async graph execution
- Mixed sync/async nodes
- Parallel execution (planned)

## Error Handling

Multi-layered error handling approach:

### 1. PocketFlow Level
- Automatic retries with configurable delays
- Fallback mechanisms
- Exception propagation control

### 2. GraphFlow Level
- State validation
- Routing error detection
- Graceful degradation

### 3. User Level
- Custom error handlers
- State-based error recovery
- Business logic error handling

## Performance Characteristics

### Startup Performance
- **Graph construction**: < 1ms
- **Compilation**: < 10ms
- **Total startup**: < 10ms

### Runtime Performance
- **Node execution**: Direct function calls
- **State updates**: Simple dictionary operations
- **Routing**: Minimal overhead
- **Memory**: Shared state, minimal copying

### Scalability
- **Node count**: No practical limits
- **State size**: Limited by available memory
- **Execution depth**: Limited by Python recursion
- **Async**: Full async/await support

## Extension Points

GraphFlow is designed to be extensible:

### 1. Custom Node Types
```python
class CustomGraphNode(GraphNode):
    def custom_processing(self, state):
        # Add custom logic
        pass
```

### 2. Custom Routing
```python
class CustomConditionalEdge(ConditionalEdge):
    def enhanced_routing(self, state):
        # Add sophisticated routing logic
        pass
```

### 3. State Processors
```python
def custom_state_merger(old_state, updates):
    # Custom state update logic
    pass
```

### 4. Execution Hooks
```python
class InstrumentedStateFlow(StateFlow):
    def _orch(self, shared, params=None):
        # Add logging, metrics, debugging
        return super()._orch(shared, params)
```

## Comparison with Other Frameworks

### vs Raw PocketFlow
- **Added**: State management, conditional routing, LangGraph-like API
- **Kept**: Simplicity, performance, minimalism
- **Trade-off**: Slightly more complexity for much more functionality

### vs LangGraph
- **Simplified**: Removed LangChain dependencies, complex reducers
- **Focused**: Core agent patterns only
- **Performance**: Faster startup, lower memory usage
- **Trade-off**: Fewer features for better simplicity

## Design Decisions

### Why Embed PocketFlow?
1. **Simplicity**: Single file distribution
2. **Control**: Can modify as needed
3. **Performance**: Direct integration
4. **Reliability**: No dependency management

### Why State-Centric?
1. **Clarity**: Easy to understand data flow
2. **Debugging**: State visible at each step
3. **Flexibility**: Nodes can read any state
4. **Composability**: Easy to combine nodes

### Why Function-Based Nodes?
1. **Simplicity**: No class hierarchy to learn
2. **Testing**: Easy to unit test
3. **Reusability**: Functions are composable
4. **Performance**: Direct function calls

This architecture provides a clean, performant foundation for building agent workflows while maintaining the simplicity that makes GraphFlow approachable and maintainable.
