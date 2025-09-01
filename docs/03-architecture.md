# Architecture Guide: How GraphFlow Works

*Understanding the parallel execution engine and core architecture*

## 🏗️ High-Level Architecture

GraphFlow is built on a **layered architecture** that transforms simple node functions into a sophisticated parallel execution system:

```
┌─────────────────────────────────────────────────────────────┐
│                  GraphFlow API Layer                        │
│         StateGraph(), Command, StateGraph                   │
├─────────────────────────────────────────────────────────────┤
│              Parallel Execution Engine                      │
│    ParallelGraphExecutor, State, Dependency Manager         │
├─────────────────────────────────────────────────────────────┤
│                Enhanced Node Layer                          │
│         Command, State Reducers, Graph Builder              │
├─────────────────────────────────────────────────────────────┤
│              Core Framework                                 │
│        StateGraph, CompiledStateGraph (Pure Python)         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 The Paradigm Shift: Linear vs Parallel

### Old Architecture: Linear Execution
The original GraphFlow used a simple **while loop** approach:

```python
# OLD: Linear execution (simplified)
def execute_graph(start_node, state):
    current_node = start_node
    
    while current_node:
        result = current_node.run(state)            # Execute one node
        next_node = determine_next(current_node)    # Find next node
        current_node = next_node                    # Move to next
    
    return state
```

**Problems:**
- ❌ Only one node executes at a time
- ❌ No parallelism possible  
- ❌ Cannot handle fan-out/fan-in patterns
- ❌ Inefficient for independent tasks

### New Architecture: Parallel Graph Traversal

GraphFlow uses a **sophisticated scheduler** that manages multiple concurrent executions:

```python
# NEW: Parallel execution (simplified)
async def execute_graph(graph, initial_state):
    state = State(initial_state)
    completed_nodes = set()
    
    # Find all nodes ready to execute (no unmet dependencies)
    ready_nodes = find_entry_nodes(graph)
    
    while ready_nodes:
        # Execute ALL ready nodes in parallel
        running_tasks = [execute_node(node, state) for node in ready_nodes]
        completed_tasks = await asyncio.gather(*running_tasks)
        
        # Process results and find newly ready nodes
        for task in completed_tasks:
            merge_result_into_state(state, task.result)
            completed_nodes.add(task.node_name)
            ready_nodes.extend(find_newly_ready_nodes(task.node_name, completed_nodes))
    
    return state
```

**Benefits:**
- ✅ Multiple nodes execute simultaneously
- ✅ True parallelism with proper synchronization
- ✅ Handles complex dependency graphs
- ✅ Maximizes resource utilization

## 🧠 Core Components Deep Dive

### 1. State Management System

The **State** class is the heart of GraphFlow's concurrent state management:

```python
class State(dict):
    """Intelligent state container with reducer-based merging"""
    
    def __init__(self, initial_data, reducers=None):
        super().__init__(initial_data)
        self.reducers = reducers or {}
        self._setup_default_reducers()
    
    def merge(self, update, field_reducers=None):
        """Merge updates using field-specific reduction strategies"""
        for key, value in update.items():
            reducer = self._choose_reducer(key, field_reducers)
            self[key] = reducer(self.get(key), value)
        return self
```

**Key Features:**
- **Reducer Functions**: Define how fields merge (extend, append, merge, set)
- **Type-Aware**: Automatically chooses appropriate reducers
- **Thread-Safe**: Handles concurrent updates safely
- **Deep Copy**: Maintains isolation between parallel branches

### 2. Parallel Execution Engine

The **ParallelGraphExecutor** orchestrates the entire execution:

```python
class ParallelGraphExecutor:
    """Core engine that manages parallel node execution"""
    
    async def ainvoke(self, initial_state, field_reducers=None):
        state = State(initial_state)
        completed_nodes = set()
        active_executions = {}
        
        # Main execution loop
        while True:
            # 1. Find all nodes ready to execute
            ready_nodes = self._find_ready_nodes(completed_nodes)
            
            if not ready_nodes and not active_executions:
                break  # No more work to do
            
            # 2. Start all ready nodes in parallel
            for node_name in ready_nodes:
                task = asyncio.create_task(self._execute_node(node_name, state))
                active_executions[node_name] = task
            
            # 3. Wait for at least one to complete
            done, pending = await asyncio.wait(
                active_executions.values(), 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 4. Process completed nodes
            for task in done:
                node_name = self._get_node_name_for_task(task)
                result = await task
                
                # Merge result into global state
                state = self._merge_result(state, result, field_reducers)
                
                # Mark as completed
                completed_nodes.add(node_name)
                del active_executions[node_name]
        
        return state
```

**Key Features:**
- **Async/Await**: Native asynchronous execution
- **Dependency Tracking**: Only runs nodes when dependencies are satisfied  
- **State Synchronization**: Thread-safe merging of concurrent updates
- **Error Handling**: Graceful handling of node failures

### 3. Dependency Management

The engine builds a **dependency graph** to track execution order:

```python
def _build_topology(self):
    """Analyze graph structure to build dependency relationships"""
    self.successors = defaultdict(list)  # node -> [successor_nodes]
    self.predecessors = defaultdict(list)  # node -> [predecessor_nodes]
    
    # Process direct edges
    for from_node, to_node in self.graph.edges.items():
        self.successors[from_node].append(to_node)
        self.predecessors[to_node].append(from_node)
    
    # Handle command-based routing dynamically
    self.conditional_nodes = set(self.graph.conditional_edges.keys())

def are_dependencies_met(self, node_name, completed_nodes):
    """Check if all predecessors have completed"""
    required_predecessors = set(self.predecessors.get(node_name, []))
    return required_predecessors.issubset(completed_nodes)
```

### 4. Command System

**Commands** enable dynamic routing with state updates:

```python
@dataclass
class Command:
    """Combines state updates with routing decisions"""
    update: Optional[Dict[str, Any]] = None     # State updates to apply
    goto: Optional[Union[str, List[str]]] = None # Next node(s) to execute
    resume: Optional[Dict[str, Any]] = None     # For future checkpointing

# Usage in nodes
def dynamic_router(state):
    if state['confidence'] > 0.9:
        return Command(
            update={'status': 'high_confidence'},
            goto='final_processor'                # Single route
        )
    else:
        return Command(
            update={'status': 'needs_review'},
            goto=['human_reviewer', 'extra_analysis']  # Fan-out!
        )
```

## 🔄 Execution Flow Walkthrough

Let's trace through how a parallel graph executes:

### Example Graph
```
       start
      /  |  \
     A   B   C
      \ | /
      combine
```

### Step-by-Step Execution

1. **Initialization**
   ```python
   ready_nodes = ['start']  # Entry point
   completed_nodes = set()
   active_executions = {}
   ```

2. **Round 1: Execute 'start'**
   ```python
   # start node returns Command(goto=['A', 'B', 'C'])
   completed_nodes = {'start'}
   ready_nodes = ['A', 'B', 'C']  # All have dependency 'start' satisfied
   ```

3. **Round 2: Execute A, B, C in parallel**
   ```python
   # All three nodes execute simultaneously
   active_executions = {
       'A': task_A,
       'B': task_B, 
       'C': task_C
   }
   ```

4. **Round 3: Results synchronization**
   ```python
   # As each completes, results are merged into state
   # State grows: {'results': ['A done', 'B done', 'C done']}
   completed_nodes = {'start', 'A', 'B', 'C'}
   ```

5. **Round 4: Execute 'combine'**
   ```python
   # combine dependencies: {'A', 'B', 'C'} ⊆ {'start', 'A', 'B', 'C'} ✓
   ready_nodes = ['combine']
   ```

6. **Completion**
   ```python
   # combine completes, no more ready nodes
   execution_complete = True
   ```

## ⚙️ Engine Configuration

### Execution Modes

GraphFlow supports multiple execution modes:

```python
# Parallel execution (default) - New engine
app = graph.compile(use_parallel_engine=True)

# Linear execution - Legacy compatibility  
app = graph.compile(use_parallel_engine=False)

# Configure concurrency limits
app = graph.compile(use_parallel_engine=True, max_concurrent=5)
```

### State Reducer Configuration

Configure how concurrent updates merge:

```python
graph = StateGraph(state_reducers={
    'results': 'extend',       # Lists: extend with new items
    'metadata': 'merge',       # Dicts: merge keys
    'counter': 'append',       # Values: append to list  
    'status': 'set'            # Values: replace with newest
})
```

### Custom Reducers

Define your own merging logic:

```python
def custom_reducer(current_value, new_value):
    """Custom logic for merging values"""
    if current_value is None:
        return new_value
    return f"{current_value} + {new_value}"

graph = StateGraph(state_reducers={
    'custom_field': custom_reducer
})
```

## 🔍 Graph Analysis Tools

GraphFlow includes built-in analysis capabilities:

### Topology Analysis
```python
compiled_graph = graph.compile()
analyzer = compiled_graph.analyze_graph()

# Check for problems
cycles = analyzer.detect_cycles()
unreachable = analyzer.find_unreachable_nodes()

if cycles:
    print(f"Warning: Found {len(cycles)} cycles")
if unreachable:
    print(f"Warning: {len(unreachable)} unreachable nodes")
```

### Performance Monitoring
```python
import time

# Time execution
start = time.time()
result = app.invoke(initial_state)
execution_time = time.time() - start

print(f"Execution completed in {execution_time:.2f}s")
print(f"Final state: {len(result)} fields")
```

## 🔧 Advanced Configuration

### Error Handling

Configure how the engine handles node failures:

```python
class ResilientGraphNode(GraphNode):
    def __init__(self, func, max_retries=3, fallback=None):
        super().__init__(func)
        self.max_retries = max_retries
        self.fallback = fallback
    
    async def execute_with_retry(self, state):
        for attempt in range(self.max_retries):
            try:
                return await self.func(state)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    if self.fallback:
                        return self.fallback(state, e)
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Custom Scheduling

For advanced use cases, you can customize the scheduling logic:

```python
class PriorityGraphExecutor(ParallelGraphExecutor):
    def _find_ready_nodes(self, completed_nodes):
        """Custom scheduling with priority support"""
        ready = super()._find_ready_nodes(completed_nodes)
        
        # Sort by priority (higher priority first)
        return sorted(ready, key=lambda node: self.get_node_priority(node), reverse=True)
    
    def get_node_priority(self, node_name):
        return self.graph.nodes[node_name].priority if hasattr(self.graph.nodes[node_name], 'priority') else 0
```

## 🎯 Design Principles

GraphFlow follows key design principles:

### 1. **Simplicity Over Complexity**
- Clear, readable code
- Minimal API surface
- Easy to understand and debug

### 2. **Performance by Default** 
- Parallel execution as the default mode
- Efficient dependency tracking
- Minimal overhead

### 3. **Developer Experience**
- Type hints throughout
- Clear error messages
- Rich debugging support

### 4. **Backward Compatibility**
- Legacy linear execution still supported
- Existing code continues to work
- Gradual migration path

## 🚀 What's Next?

Now that you understand the architecture, explore:

1. **[Parallel Patterns](04-parallel-patterns.md)** - Common execution patterns
2. **[State Management](05-state-management.md)** - Advanced state handling
3. **[Building Workflows](06-building-workflows.md)** - Complex workflow construction
4. **[API Reference](07-api-reference.md)** - Complete API documentation

The architecture of GraphFlow enables sophisticated parallel AI agent workflows while maintaining the simplicity that made the original GraphFlow appealing. Build with confidence! 🎉

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

### Node Functions

Simple Python functions that process state:

```python
def process_data(state):
    """Simple node function that processes state."""
    data = state.get('input_data', '')
    processed = f"Processed: {data}"
    return {'output': processed}

def conditional_router(state):
    """Node that routes based on state conditions."""
    if state.get('urgent', False):
        return Command(goto='priority_handler')
    else:
        return Command(goto='normal_handler')

def parallel_distributor(state):
    """Fan-out to multiple parallel processors."""
    return Command(
        update={'status': 'processing'},
        goto=['processor_a', 'processor_b', 'processor_c']
    )
```

**Key Features:**
- Simple Python functions - no classes required
- Receive full state as input
- Return state updates or Command objects
- Support both sync and async execution

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

## 🎓 Key Takeaways

GraphFlow delivers a powerful parallel execution architecture that's both simple to use and sophisticated under the hood:

### ✅ **For Developers**
- **Simple API** - Create parallel workflows with minimal code
- **Zero Dependencies** - Pure Python, no external requirements
- **Auto-Detection** - Parallel engine activates automatically when beneficial
- **Backward Compatible** - Linear execution still available

### ✅ **For Performance**
- **True Parallelism** - Multiple nodes execute simultaneously
- **Smart State Merging** - Conflict-free concurrent state updates
- **Efficient Scheduling** - Dependency-aware execution order
- **Minimal Overhead** - Lightweight execution engine

### ✅ **For Scale**
- **Fan-out/Fan-in** - Natural patterns for parallel processing
- **Resource Management** - Controlled concurrency levels
- **Error Resilience** - Robust handling of node failures
- **Production Ready** - Monitoring and debugging capabilities

**GraphFlow proves that powerful parallel execution doesn't require complex frameworks - sometimes the best architecture is the simplest one that works.** 🚀
