# API Reference: Complete Guide to GraphFlow Functions and Classes

*Comprehensive reference for all GraphFlow APIs, classes, and functions*

## 📚 Overview

This reference covers all public APIs in GraphFlow, organized by category:

1. **Core Graph API** - Building and executing graphs
2. **Parallel Engine** - Advanced parallel execution features  
3. **State Management** - State classes and reducers
4. **Command System** - Flow control and routing
5. **Utilities** - Helper functions and tools

## 🏗️ Core Graph API

### `create_graph(state_reducers=None)`

Creates a new graph builder for constructing workflows.

**Parameters:**
- `state_reducers` (dict, optional): State field reducers for parallel execution
  - Keys: State field names (strings)
  - Values: Reducer type ('extend', 'append', 'merge', 'set') or custom function

**Returns:** `StateGraph` - Graph builder instance

**Example:**
```python
from graphflow import create_graph

# Basic graph
graph = StateGraph()

# Graph with state reducers for parallel execution
graph = StateGraph(state_reducers={
    'results': 'extend',      # Combine lists from parallel nodes
    'metadata': 'merge',      # Merge dictionaries  
    'status': 'set',          # Last write wins
    'logs': 'extend'          # Collect logs from all nodes
})
```

---

### `StateGraph` Class

Graph builder for constructing workflows.

#### `add_node(name, func)`

Add a processing node to the graph.

**Parameters:**
- `name` (str): Unique node identifier
- `func` (callable): Node function that processes state

**Returns:** `StateGraph` - Self for method chaining

**Node Function Signature:**
```python
def node_function(state: dict) -> Union[dict, Command]:
    """
    Process state and return updates or routing commands.
    
    Args:
        state: Current workflow state
        
    Returns:
        dict: State updates to apply
        Command: State updates + routing instructions
    """
```

**Example:**
```python
def process_data(state):
    data = state.get('input_data', '')
    processed = f"Processed: {data}"
    return {'output': processed}

graph.add_node('processor', process_data)
```

#### `add_edge(from_node, to_node)`

Add a directed edge between nodes.

**Parameters:**
- `from_node` (str): Source node name
- `to_node` (str): Target node name  

**Returns:** `StateGraph` - Self for method chaining

**Example:**
```python
graph.add_edge('input_processor', 'data_analyzer')
```

#### `set_entry_point(node_name)`

Set the starting node for graph execution.

**Parameters:**
- `node_name` (str): Name of the entry node

**Returns:** `StateGraph` - Self for method chaining

**Example:**
```python
graph.set_entry_point('start_node')
```

#### `compile(use_parallel_engine=None, **kwargs)`

Compile the graph into an executable workflow.

**Parameters:**
- `use_parallel_engine` (bool, optional): Force parallel/linear execution mode
  - `True`: Use parallel engine (requires engine.py)
  - `False`: Use linear execution  
  - `None`: Auto-detect based on state_reducers and engine availability

**Returns:** `CompiledStateGraph` - Executable workflow

**Example:**
```python
# Auto-detect execution mode
app = graph.compile()

# Force parallel execution
app = graph.compile(use_parallel_engine=True)

# Force linear execution  
app = graph.compile(use_parallel_engine=False)
```

---

### `CompiledStateGraph` Class

Executable workflow that can process state through the graph.

#### `invoke(initial_state, **kwargs)`

Execute the workflow with initial state.

**Parameters:**
- `initial_state` (dict): Starting state for the workflow
- `**kwargs`: Additional execution parameters

**Returns:** `dict` - Final state after workflow execution

**Example:**
```python
app = graph.compile()
result = app.invoke({'input': 'Hello World'})
print(result)  # Final state with all processing results
```

#### `ainvoke(initial_state, **kwargs)` (Async)

Asynchronous execution of the workflow.

**Parameters:**
- `initial_state` (dict): Starting state for the workflow  
- `**kwargs`: Additional execution parameters

**Returns:** `dict` - Final state after workflow execution

**Example:**
```python
import asyncio

app = graph.compile()
result = await app.ainvoke({'input': 'Hello World'})
```

---

## ⚡ Parallel Engine API

### `State` Class

Enhanced state management for parallel execution with reducer support.

#### `__init__(data=None, reducers=None)`

Create a new State instance.

**Parameters:**
- `data` (dict, optional): Initial state data
- `reducers` (dict, optional): Field-specific merge strategies

#### `merge(updates)`

Merge state updates using configured reducers.

**Parameters:**
- `updates` (dict): State updates to merge

**Example:**
```python
from engine import State

state = State(
    data={'results': ['item1']},
    reducers={'results': 'extend'}
)

state.merge({'results': ['item2', 'item3']})
print(state.data['results'])  # ['item1', 'item2', 'item3']
```

#### `get(key, default=None)`

Get state value with optional default.

**Parameters:**
- `key` (str): State field name
- `default`: Default value if key not found

**Returns:** State field value or default

#### `set(key, value)`

Set state field value directly.

**Parameters:**
- `key` (str): State field name  
- `value`: Value to set

---

### `ParallelGraphExecutor` Class

Advanced parallel execution engine.

#### `__init__(graph, reducers=None)`

Create parallel executor for a graph.

**Parameters:**
- `graph`: Graph structure with nodes and edges
- `reducers` (dict, optional): State field reducers

#### `ainvoke(initial_state)`

Execute graph with parallel node processing.

**Parameters:**
- `initial_state` (dict): Starting state

**Returns:** `dict` - Final execution state

**Features:**
- ✅ Automatic dependency resolution
- ✅ Concurrent node execution  
- ✅ State merging with reducers
- ✅ Deadlock detection
- ✅ Fan-out/fan-in support

---

## 🎛️ Command System API

### `Command` Class

Flow control object for node routing and state updates.

#### `__init__(update=None, goto=None)`

Create a command for state updates and routing.

**Parameters:**
- `update` (dict, optional): State updates to apply
- `goto` (str or list, optional): Next node(s) to execute

**Example:**
```python
from graphflow import Command

# State update only
return Command(update={'status': 'processed'})

# Routing only  
return Command(goto='next_node')

# Both update and routing
return Command(
    update={'result': 'success'},
    goto='completion_handler'
)

# Fan-out to multiple nodes
return Command(
    update={'status': 'distributing'},
    goto=['worker1', 'worker2', 'worker3']
)
```

#### Properties

- `update` (dict): State updates to apply
- `goto` (str or list): Target node(s) for routing

---

## 🔧 State Reducers Reference

### Built-in Reducer Types

#### `'extend'` - List Concatenation
Combines lists from multiple sources.

**Behavior:**
```python
current: ['a', 'b']
new: ['c', 'd']  
result: ['a', 'b', 'c', 'd']
```

**Use cases:** Collecting parallel results, accumulating logs

#### `'append'` - Single Item Addition  
Adds individual items to a list.

**Behavior:**
```python
current: ['a', 'b']
new: 'c'
result: ['a', 'b', 'c']
```

**Use cases:** Adding single values, event logging

#### `'merge'` - Dictionary Merging
Combines dictionary fields.

**Behavior:**
```python
current: {'x': 1, 'y': 2}
new: {'y': 3, 'z': 4}
result: {'x': 1, 'y': 3, 'z': 4}  # new values override
```

**Use cases:** Metadata collection, configuration objects

#### `'set'` - Value Replacement
Replaces value with latest update.

**Behavior:**
```python
current: 'old_value'
new: 'new_value'
result: 'new_value'
```

**Use cases:** Status updates, final results

### Custom Reducer Functions

Create custom merge logic with reducer functions:

```python
def custom_reducer(current_value, new_value):
    """
    Custom state merge logic.
    
    Args:
        current_value: Existing state value
        new_value: New value to merge
        
    Returns:
        Merged value
    """
    # Your custom merge logic here
    return merged_value

# Use in graph
graph = StateGraph(state_reducers={
    'custom_field': custom_reducer
})
```

**Examples:**

```python
# Priority-based selection
def priority_reducer(current, new):
    if not current:
        return new
    if not new:
        return current
    return new if new.get('priority', 0) > current.get('priority', 0) else current

# Numeric accumulation
def sum_reducer(current, new):
    return (current or 0) + (new or 0)

# Smart list merging (no duplicates)  
def unique_extend_reducer(current, new):
    current_list = current or []
    new_list = new if isinstance(new, list) else [new]
    return current_list + [item for item in new_list if item not in current_list]
```

---

## 🛠️ Utility Functions

### Graph Analysis

#### `get_dependencies(graph)`
Analyze node dependencies in a graph.

**Parameters:**
- `graph`: Graph structure  

**Returns:** `dict` - Node dependency mapping

#### `detect_cycles(graph)`
Check for circular dependencies.

**Parameters:**
- `graph`: Graph structure

**Returns:** `list` - Detected cycles (empty if none)

#### `topological_sort(graph)`
Get execution order respecting dependencies.

**Parameters:**
- `graph`: Graph structure

**Returns:** `list` - Nodes in dependency order

---

## 🔍 Error Handling

### Exception Types

#### `GraphExecutionError`
Raised when graph execution fails.

**Attributes:**
- `message` (str): Error description
- `node` (str): Node where error occurred
- `state` (dict): State at time of error

#### `InvalidGraphError`  
Raised when graph structure is invalid.

**Attributes:**
- `message` (str): Error description
- `issues` (list): Specific validation issues

#### `StateReducerError`
Raised when state reducer fails.

**Attributes:**
- `message` (str): Error description
- `field` (str): State field being reduced
- `reducer` (str): Reducer that failed

### Error Handling Patterns

```python
from graphflow import GraphExecutionError

def robust_node(state):
    try:
        result = risky_operation(state)
        return {'result': result}
    except Exception as e:
        # Handle error gracefully
        return {
            'error': str(e),
            'status': 'failed'
        }

# Or use Command for error routing
def node_with_error_routing(state):
    try:
        result = risky_operation(state)
        return Command(
            update={'result': result},
            goto='success_handler'
        )
    except Exception as e:
        return Command(
            update={'error': str(e)},
            goto='error_handler'
        )
```

---

## ⚙️ Configuration Options

### Environment Variables

#### `GRAPHFLOW_DEBUG`
Enable debug logging.
```bash
export GRAPHFLOW_DEBUG=1
```

#### `GRAPHFLOW_PARALLEL_DEFAULT`
Set default parallel execution preference.
```bash
export GRAPHFLOW_PARALLEL_DEFAULT=true
```

### Compilation Options

```python
app = graph.compile(
    use_parallel_engine=True,     # Force parallel execution
    max_parallel_nodes=4,         # Limit concurrent nodes
    timeout=30,                   # Execution timeout (seconds)
    debug=True                    # Enable debug output
)
```

---

## 📊 Performance Considerations

### Parallel vs Linear Execution

**Use Parallel When:**
- ✅ Multiple independent processing steps
- ✅ I/O-bound operations (API calls, file operations)
- ✅ CPU-intensive tasks that can run concurrently
- ✅ Fan-out/fan-in patterns

**Use Linear When:**
- ✅ Simple sequential workflows
- ✅ Tight memory constraints
- ✅ Debugging complex workflows
- ✅ Maximum simplicity needed

### Performance Monitoring

```python
import time

def timed_node(state):
    start = time.time()
    result = process_data(state)
    duration = time.time() - start
    
    return {
        'result': result,
        'performance': {
            'node': 'data_processor',
            'duration': duration
        }
    }
```

---

## 🎯 Complete Example

Here's a comprehensive example showing most API features:

```python
from graphflow import StateGraph, Command
import time

def create_advanced_workflow():
    """Complete example with all API features"""
    
    # Create graph with state reducers
    graph = StateGraph(state_reducers={
        'results': 'extend',        # Collect parallel results
        'metadata': 'merge',        # Combine metadata
        'final_status': 'set',      # Status updates
        'processing_log': 'extend', # Execution log
        'errors': 'extend'          # Error collection
    })
    
    def input_validator(state):
        """Validate input and route accordingly"""
        data = state.get('input_data', '')
        
        if not data:
            return Command(
                update={'errors': ['No input data provided']},
                goto='error_handler'
            )
        
        return Command(
            update={
                'validated_input': data,
                'processing_log': ['Input validation completed']
            },
            goto=['processor_a', 'processor_b', 'processor_c']  # Fan-out
        )
    
    def processor(name, delay=0.1):
        """Parallel processor with timing"""
        def process_func(state):
            start_time = time.time()
            
            # Simulate processing
            time.sleep(delay)
            input_data = state.get('validated_input', '')
            result = f'{name} processed: {input_data}'
            
            return {
                'results': [result],
                'metadata': {
                    f'{name}_duration': time.time() - start_time,
                    f'{name}_timestamp': time.time()
                },
                'processing_log': [f'{name} completed']
            }
        return process_func
    
    def result_combiner(state):
        """Combine parallel results"""
        results = state.get('results', [])
        metadata = state.get('metadata', {})
        logs = state.get('processing_log', [])
        
        return {
            'final_status': 'completed',
            'processing_log': ['Result combination completed'],
            'summary': {
                'total_results': len(results),
                'processing_steps': len(logs),
                'execution_metadata': metadata,
                'combined_results': results
            }
        }
    
    def error_handler(state):
        """Handle any errors that occurred"""
        errors = state.get('errors', [])
        return {
            'final_status': 'error',
            'error_summary': {
                'error_count': len(errors),
                'errors': errors
            }
        }
    
    # Build the graph
    (graph
     .add_node('validator', input_validator)
     .add_node('processor_a', processor('ProcessorA', 0.1))
     .add_node('processor_b', processor('ProcessorB', 0.15))
     .add_node('processor_c', processor('ProcessorC', 0.12))
     .add_node('combiner', result_combiner)
     .add_node('error_handler', error_handler)
     .set_entry_point('validator')
     
     # Fan-in: all processors → combiner
     .add_edge('processor_a', 'combiner')
     .add_edge('processor_b', 'combiner')
     .add_edge('processor_c', 'combiner'))
    
    return graph.compile()

# Usage example
if __name__ == "__main__":
    # Create and test the workflow
    workflow = create_advanced_workflow()
    
    # Test with valid input
    result = workflow.invoke({'input_data': 'test data'})
    print("✅ Success case:")
    print(f"Status: {result['final_status']}")
    print(f"Results: {len(result['summary']['combined_results'])}")
    print(f"Steps: {result['summary']['processing_steps']}")
    
    # Test with invalid input
    error_result = workflow.invoke({})  # No input_data
    print("\n❌ Error case:")
    print(f"Status: {error_result['final_status']}")
    print(f"Errors: {error_result['error_summary']['errors']}")
```

This example demonstrates:
- ✅ Graph creation with state reducers
- ✅ Input validation and conditional routing
- ✅ Parallel processing with fan-out/fan-in
- ✅ Error handling and routing
- ✅ Performance monitoring
- ✅ State merging with multiple reducer types
- ✅ Complete workflow compilation and execution

---

## 🎓 Next Steps

1. **Explore Examples** - Check the `examples/` directory for more patterns
2. **Read Documentation** - Review other docs for detailed explanations
3. **Build Your Own** - Start with simple workflows and add complexity
4. **Contribute** - Help improve GraphFlow with feedback and contributions

**Happy building with GraphFlow!** 🚀
