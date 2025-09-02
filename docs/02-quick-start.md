# Quick Start Guide

*Get up and running with GraphFlow parallel execution in 5 minutes*

## ⚡ Installation

GraphFlow has **zero dependencies** - just Python 3.7+:

```bash
git clone https://github.com/zhwa/GraphFlow.git
cd GraphFlow
python examples/01-test-and-examples.py  # Verify everything works
```

You should see:
```
✅ GraphFlow imported successfully
✅ Parallel execution successful
✅ Parallel fan-out/fan-in test successful
```

## 🎯 Your First Parallel Graph

Let's build a simple parallel workflow that processes data with multiple workers:

```python
from graphflow import StateGraph, Command
import time

# 1. Create a graph with smart state management
graph = StateGraph(state_reducers={
    'results': 'extend',  # Automatically merge results from parallel nodes
    'timings': 'extend'   # Collect timing data
})

# 2. Define your workflow nodes
def start_work(state):
    """Entry point that fans out to parallel workers"""
    print("🚀 Starting parallel work...")
    
    return Command(
        update={'started_at': time.time()},
        goto=['worker_a', 'worker_b', 'worker_c']  # Fan-out to 3 parallel workers!
    )

def worker_a(state):
    """Worker A - does some processing"""
    print("⚙️ Worker A processing...")
    time.sleep(0.5)  # Simulate work
    
    return {
        'results': ['Worker A completed task'],
        'timings': [{'worker_a': time.time()}]
    }

def worker_b(state):
    """Worker B - does different processing (runs in parallel with A and C)"""
    print("🔧 Worker B processing...")
    time.sleep(0.3)  # Different timing
    
    return {
        'results': ['Worker B completed task'],
        'timings': [{'worker_b': time.time()}]
    }

def worker_c(state):
    """Worker C - yet another parallel process"""
    print("⚡ Worker C processing...")
    time.sleep(0.7)  # Longest task
    
    return {
        'results': ['Worker C completed task'],
        'timings': [{'worker_c': time.time()}]
    }

def combine_results(state):
    """Fan-in: Combines all parallel results"""
    print("🔗 Combining results from all workers...")
    
    num_results = len(state.get('results', []))
    total_time = time.time() - state.get('started_at', time.time())
    
    return {
        'final_report': f"Successfully processed {num_results} parallel tasks in {total_time:.2f}s",
        'status': 'completed'
    }

# 3. Build the graph structure
(graph
 .add_node('start', start_work)
 .add_node('worker_a', worker_a)
 .add_node('worker_b', worker_b)
 .add_node('worker_c', worker_c)
 .add_node('combiner', combine_results)
 .set_entry_point('start')
 
 # Fan-in: All workers feed into combiner
 .add_edge('worker_a', 'combiner')
 .add_edge('worker_b', 'combiner')
 .add_edge('worker_c', 'combiner'))

# 4. Compile and run with parallel execution
app = graph.compile(use_parallel_engine=True)

print("Starting parallel execution...")
start_time = time.time()

result = app.invoke({'input': 'test data'})

end_time = time.time()

# 5. See the results!
print(f"\n✅ {result['final_report']}")
print(f"Results collected: {result.get('results', [])}")
```

## � Example 2: Conditional Routing

Here's a more advanced example with conditional logic:

```python
from graphflow import StateGraph, Command

def create_data_processor():
    graph = StateGraph(state_reducers={'outputs': 'extend'})
    
    def classifier(state):
        """Route to different processors based on data type"""
        data_type = state.get('data_type', 'unknown')
        
        if data_type == 'text':
            return Command(
                update={'classification': 'text_processing'},
                goto=['nlp_processor', 'sentiment_analyzer']  # Parallel text processing
            )
        elif data_type == 'image':
            return Command(
                update={'classification': 'image_processing'}, 
                goto=['object_detector', 'image_enhancer']    # Parallel image processing
            )
        else:
            return Command(
                update={'classification': 'generic_processing'},
                goto='generic_processor'                      # Single fallback
            )
    
    def nlp_processor(state):
        return {'outputs': ['NLP analysis complete']}
    
    def sentiment_analyzer(state):
        return {'outputs': ['Sentiment: positive (0.8)']}
    
    def object_detector(state):
        return {'outputs': ['Found 3 objects: cat, tree, car']}
    
    def image_enhancer(state):
        return {'outputs': ['Image enhanced: +20% quality']}
    
    def generic_processor(state):
        return {'outputs': ['Generic processing complete']}
    
    def final_aggregator(state):
        """Combines results regardless of path taken"""
        outputs = state.get('outputs', [])
        classification = state.get('classification', 'unknown')
        
        return {
            'summary': f"{classification}: {len(outputs)} tasks completed",
            'all_outputs': outputs
        }
    
    # Build the graph
    (graph
     .add_node('classifier', classifier)
     .add_node('nlp_processor', nlp_processor)
     .add_node('sentiment_analyzer', sentiment_analyzer)
     .add_node('object_detector', object_detector)
     .add_node('image_enhancer', image_enhancer)
     .add_node('generic_processor', generic_processor)
     .add_node('aggregator', final_aggregator)
     .set_entry_point('classifier')
     
     # All paths lead to aggregator
     .add_edge('nlp_processor', 'aggregator')
     .add_edge('sentiment_analyzer', 'aggregator')
     .add_edge('object_detector', 'aggregator')
     .add_edge('image_enhancer', 'aggregator')
     .add_edge('generic_processor', 'aggregator'))
    
    return graph.compile()

# Test different input types
processor = create_data_processor()

# Text processing (2 parallel tasks)
result = processor.invoke({'data_type': 'text'})
print(f"Text: {result['summary']}")

# Image processing (2 parallel tasks)  
result = processor.invoke({'data_type': 'image'})
print(f"Image: {result['summary']}")

# Unknown type (single task)
result = processor.invoke({'data_type': 'unknown'})
print(f"Unknown: {result['summary']}")
```

## 🚀 What's Next?

You now have the basics! Explore more advanced topics:

- **[Core Concepts](01-core-concepts.md)** - Understand parallel execution fundamentals
- **[Parallel Patterns](03-parallel-patterns.md)** - Common workflow patterns
- **[State Management](04-state-management.md)** - Advanced state handling
- **[Building Workflows](05-building-workflows.md)** - Complex real-world examples
- **[Examples Directory](../examples/)** - More working code samples

## 💡 Key Points to Remember

1. **Fan-out with Command**: Use `goto=['node1', 'node2', 'node3']` to trigger parallel execution
2. **Fan-in with edges**: Multiple nodes pointing to the same target create synchronization points
3. **State reducers**: Configure how parallel updates merge (`'extend'`, `'append'`, `'merge'`, `'set'`)
4. **Conditional routing**: Commands can route to different nodes based on state
5. **Zero configuration**: Parallel execution happens automatically when you use the patterns above
     .set_entry_point('start')
     .add_edge('task1', 'combiner')
     .add_edge('task2', 'combiner')
     .add_edge('task3', 'combiner')
     .add_edge('task4', 'combiner'))
    
    return graph

# Test parallel execution
print("🚀 Testing parallel execution...")
parallel_graph = create_performance_test().compile(use_parallel_engine=True)
start = time.time()
result = parallel_graph.invoke({'input': 'test'})
parallel_time = time.time() - start

# Test linear execution
print("🐌 Testing linear execution...")
linear_graph = create_performance_test().compile(use_parallel_engine=False)
start = time.time()
result = linear_graph.invoke({'input': 'test'})
linear_time = time.time() - start

print(f"\n📊 Performance Results:")
print(f"Parallel execution: {parallel_time:.2f}s")
print(f"Linear execution: {linear_time:.2f}s") 
print(f"Speedup: {linear_time / parallel_time:.1f}x faster!")
```

**Expected Output:**
```
📊 Performance Results:
Parallel execution: 0.20s
Linear execution: 0.80s
Speedup: 4.0x faster!
```

## 🧩 Key Concepts in Action

From these examples, you can see the core GraphFlow concepts:

### ✅ **Fan-Out (1 → Many)**
```python
goto=['worker_a', 'worker_b', 'worker_c']  # One node triggers 3 parallel nodes
```

### ✅ **Fan-In (Many → 1)**  
```python
# All workers connect to combiner - it waits for all to complete
.add_edge('worker_a', 'combiner')
.add_edge('worker_b', 'combiner')  
.add_edge('worker_c', 'combiner')
```

### ✅ **Smart State Management**
```python
state_reducers={'results': 'extend'}  # Automatically merge parallel results
```

### ✅ **Dynamic Routing**
```python
if data_type == 'text':
    return Command(goto=['nlp_processor', 'sentiment_analyzer'])
elif data_type == 'image':
    return Command(goto=['object_detector', 'image_enhancer'])
```

## 🚀 What's Next?

Now that you've built your first parallel graphs, you're ready to:

1. **Learn advanced patterns**: [Parallel Patterns](03-parallel-patterns.md)
2. **Master state management**: [State Management](04-state-management.md)  
3. **Build complex workflows**: [Building Workflows](05-building-workflows.md)
4. **Explore real examples**: Check out the [examples/](../examples/) directory

**Try modifying the examples above and see what happens!** The best way to learn GraphFlow is by experimenting. 🧪

## 🔗 Quick Reference

### Essential Imports
```python
from graphflow import StateGraph, Command
```

### Graph Creation
```python
graph = StateGraph(state_reducers={'results': 'extend'})
```

### Node Definition  
```python
def my_node(state):
    return {'key': 'value'}  # or Command(update={...}, goto='next_node')
```

### Graph Building
```python
(graph
 .add_node('name', function)
 .add_edge('from_node', 'to_node')
 .set_entry_point('start_node'))
```

### Execution
```python
app = graph.compile()
result = app.invoke({'input': 'data'})
```

Happy building! 🎉
