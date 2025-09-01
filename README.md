# GraphFlow

**A Powerful Parallel Graph Execution Framework for AI Agents**

GraphFlow is a **true parallel graph execution engine** that rivals LangGraph's capabilities while maintaining elegant, dependency-free design.

## 🚀 Quick Start

Get running in under 2 minutes:

```bash
git clone <repository>
cd GraphFlow
python examples/01-test-and-examples.py  # Verify installation & run tests
python examples/09-parallel-execution-demo.py  # See it in action
```

## ⚡ What Makes GraphFlow Powerful

GraphFlow implements a **sophisticated parallel graph execution architecture** from the ground up:

### 🔥 **Core Breakthrough: True Parallel Execution**
- **Fan-out**: One node triggers multiple parallel workers
- **Fan-in**: Multiple nodes synchronize into a single join point  
- **Dependency Management**: Intelligent scheduling based on completion status
- **Async Support**: Native async/await for high-performance execution

### 📊 **Declarative State Management**
- **Smart Reducers**: Automatic list extension, dictionary merging, value replacement
- **Field-Specific Logic**: Configure how each state field gets updated
- **Type-Safe**: Built on Python's typing system
- **Predictable**: No more manual state mutation bugs

### 🏗️ **Graph-Native Architecture**
- **True Graph Traversal**: Replaces linear while-loops with proper DAG execution
- **Command Routing**: Dynamic path selection with list-based fan-out
- **Conditional Edges**: State-based routing decisions
- **Cycle Detection**: Built-in analysis tools for graph validation

## 🌟 Quick Preview

Here's parallel execution in action:

```python
from graphflow import StateGraph, Command

# Create graph with intelligent state reducers
graph = StateGraph(state_reducers={'results': 'extend'})

def start_parallel_work(state):
    return Command(
        update={'phase': 'processing'},
        goto=['worker1', 'worker2', 'worker3']  # 🔀 Fan-out to 3 parallel workers
    )

def worker(worker_id):
    def work_func(state):
        # Simulate parallel processing
        time.sleep(0.5)  
        return {'results': [f'Worker {worker_id} completed']}
    return work_func

def combine_results(state):
    # 🔗 Fan-in: Waits for all workers to complete
    return {'final': f"Combined {len(state['results'])} results"}

# Build the parallel graph
(graph
 .add_node('start', start_parallel_work)
 .add_node('worker1', worker(1))
 .add_node('worker2', worker(2)) 
 .add_node('worker3', worker(3))
 .add_node('combiner', combine_results)
 .set_entry_point('start')
 .add_edge('worker1', 'combiner')  # All workers → combiner
 .add_edge('worker2', 'combiner')
 .add_edge('worker3', 'combiner'))

# Execute with parallel engine (default)
app = graph.compile()
result = app.invoke({'input': 'data'})

# Workers run in parallel! 
# Combiner waits for all to complete!
# Result: {'results': ['Worker 1 completed', 'Worker 2 completed', 'Worker 3 completed'], 'final': 'Combined 3 results'}
```

**👆 This creates true parallel execution with proper synchronization in ~20 lines!**

## 📖 Documentation

Our documentation covers both fundamentals for beginners and advanced topics:

### **🎓 Fundamentals (For Beginners)**
- **[📚 Core Concepts](docs/01-core-concepts.md)** - Parallel execution, state management, fan-out/fan-in explained
- **[🚀 Quick Start Guide](docs/02-quick-start.md)** - Your first parallel graph in 5 minutes  
- **[🏗️ Architecture Guide](docs/03-architecture.md)** - How the parallel engine works

### **💻 Practical Guides**
- **[🔄 Parallel Patterns](docs/04-parallel-patterns.md)** - Fan-out, fan-in, and synchronization patterns
- **[📊 State Management](docs/05-state-management.md)** - Reducers, state schemas, and best practices
- **[🎯 Building Workflows](docs/06-building-workflows.md)** - From simple chains to complex DAGs

### **📘 Reference**
- **[📖 API Reference](docs/07-api-reference.md)** - Complete API documentation
- **[🆚 GraphFlow vs LangGraph](docs/08-comparison.md)** - Feature comparison and migration guide

### **💡 Examples**
- **[Working Examples](examples/)** - Progressive examples from basic to advanced parallel workflows

## 🎯 Choose Your Learning Path

**🔰 New to parallel execution or graph frameworks?**  
Start with [Core Concepts](docs/01-core-concepts.md) to understand the fundamentals

**💻 Ready to build?**  
Jump to [Quick Start Guide](docs/02-quick-start.md)

**� Need parallel patterns?**  
Check [Parallel Patterns](docs/04-parallel-patterns.md)

**🔍 Want working examples?**  
Browse the [Examples Directory](examples/)

**� Migrating from LangGraph?**  
See [GraphFlow vs LangGraph](docs/08-comparison.md)

## 💡 What Makes GraphFlow Special

### **🚀 Performance & Scalability**
- **True Parallelism**: Execute multiple nodes simultaneously
- **Efficient Scheduling**: Dependency-aware execution prevents blocking
- **Async Native**: Built for high-concurrency workloads
- **Lightweight**: No heavy dependencies, fast startup

### **🧠 Developer Experience**
- **Intuitive API**: Familiar patterns from LangGraph with improved simplicity
- **Type Safe**: Full TypeScript-style type hints and validation
- **Great Debugging**: Clear execution traces and state inspection
- **Backward Compatible**: Existing linear graphs still work

### **🏗️ Architecture**
- **Graph-Native**: Purpose-built for DAG execution (not a linear system)
- **Extensible**: Clean separation between execution engine and user code
- **Testable**: Easy to unit test individual nodes and integration test graphs
- **Zero Dependencies**: Only Python standard library

### **� Analysis & Validation**
- **Graph Analysis**: Cycle detection, unreachable node detection
- **Performance Monitoring**: Built-in execution timing and profiling
- **Execution Modes**: Choose between parallel and linear execution
- **Debug Support**: Comprehensive logging and state introspection

## 🏆 GraphFlow vs The Competition

| Feature | GraphFlow | LangGraph | Airflow | Prefect |
|---------|-------------|-----------|---------|---------|
| **Parallel Execution** | ✅ True parallelism | ✅ Full support | ✅ Full support | ✅ Full support |
| **Learning Curve** | 🟢 Low | 🟡 Medium | 🔴 High | 🟡 Medium |
| **Dependencies** | ✅ Zero | 🔴 Many | 🔴 Heavy | 🔴 Heavy |
| **AI Agent Focus** | ✅ Purpose-built | ✅ Purpose-built | ❌ General workflow | ❌ General workflow |
| **Setup Time** | 🟢 Instant | 🟡 Moderate | � Complex | 🟡 Moderate |
| **Code Size** | 🟢 < 1000 lines | 🔴 Thousands | 🔴 Massive | 🔴 Large |

## 🚀 Ready to Get Started?

1. **Learn the fundamentals**: [Core Concepts](docs/01-core-concepts.md)
2. **Build your first parallel graph**: [Quick Start Guide](docs/02-quick-start.md)  
3. **Explore patterns**: [Parallel Patterns](docs/04-parallel-patterns.md)
4. **See it in action**: Run `python examples/09-parallel-execution-demo.py`

**Transform your AI agents with true parallel execution!**
