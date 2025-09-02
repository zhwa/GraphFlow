# GraphFlow

**A Powerful Parallel Graph Execution Framework for AI Agents**

GraphFlow is a **true parallel graph execution engine** that rivals LangGraph's capabilities while maintaining elegant, dependency-free design.

## 🚀 Quick Start

Get running in under 2 minutes:

```bash
git clone https://github.com/zhwa/GraphFlow.git
cd GraphFlow
python examples/01-test-and-examples.py  # Verify installation & run tests
```

## ⚡ What Makes GraphFlow Powerful

### 🔥 **True Parallel Execution**
- **Fan-out**: One node triggers multiple parallel workers
- **Fan-in**: Multiple nodes synchronize into a single join point  
- **Dependency Management**: Intelligent scheduling based on completion status
- **5x Performance**: Measured parallel speedup over sequential execution

### 📊 **Smart State Management**
- **Reducers**: Automatic list extension, dictionary merging, value replacement
- **Field-Specific Logic**: Configure how each state field gets updated
- **Conflict-Free**: No more manual state mutation bugs in parallel execution

### 🏗️ **Zero Dependencies**
- **Pure Python**: No external dependencies required
- **Lightweight**: <1000 lines of code total
- **Fast Startup**: Instant execution, no framework overhead

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
# Result: {'results': ['Worker 1 completed', 'Worker 2 completed', 'Worker 3 completed'], 'final': 'Combined 3 results'}
```

**👆 This creates true parallel execution with proper synchronization in ~20 lines!**

## 📖 Documentation

### **🎓 Getting Started**
- **[🚀 Quick Start Guide](docs/02-quick-start.md)** - Your first parallel graph in 5 minutes  
- **[📚 Core Concepts](docs/01-core-concepts.md)** - Understand parallel execution fundamentals

### **💻 Building Workflows**
- **[🔄 Parallel Patterns](docs/03-parallel-patterns.md)** - Common patterns and best practices
- **[📊 State Management](docs/04-state-management.md)** - Advanced state handling with reducers
- **[🏗️ Complete Workflows](docs/05-building-workflows.md)** - From simple chains to complex AI agents

### **📚 Reference**
- **[📋 API Reference](docs/06-api-reference.md)** - Complete function and class documentation
- **[⚖️ vs LangGraph](docs/07-comparison.md)** - Feature and performance comparison

## 💡 Examples

Explore progressive examples from basic to advanced parallel workflows:
- **[examples/](examples/)** - Complete working examples with detailed explanations

## 🎯 Choose Your Learning Path

**🔰 New to parallel execution?** → [Core Concepts](docs/01-core-concepts.md)  
**💻 Ready to build?** → [Quick Start Guide](docs/02-quick-start.md)  
**🔄 Need patterns?** → [Parallel Patterns](docs/03-parallel-patterns.md)  
**🔍 Want examples?** → [Examples Directory](examples/)  
**🆚 Migrating from LangGraph?** → [Comparison Guide](docs/07-comparison.md)

## 🚀 Ready to Get Started?

1. **Learn the fundamentals**: [Core Concepts](docs/01-core-concepts.md)
2. **Build your first parallel graph**: [Quick Start Guide](docs/02-quick-start.md)  
3. **Explore patterns**: [Parallel Patterns](docs/03-parallel-patterns.md)