# Parallel Patterns: Mastering Concurrent Workflows

*Common patterns and best practices for building parallel graphs in GraphFlow*

## 🎯 Overview

This guide covers the essential patterns you'll use to build sophisticated parallel workflows. Each pattern is explained with practical examples and real-world use cases.

## 🔀 Pattern 1: Basic Fan-Out/Fan-In

The most fundamental parallel pattern - distribute work to multiple processors, then combine results.

### Structure
```
    start
   / | | \
  A  B C  D
   \ | | /
   combine
```

### Implementation
```python
from graphflow import StateGraph, Command

def create_basic_fanout():
    graph = StateGraph(state_reducers={'results': 'extend'})

    def distribute_work(state):
        """Fan-out: Send work to multiple parallel processors"""
        work_items = state.get('work_items', [])

        return Command(
            update={'status': 'processing', 'started_at': time.time()},
            goto=['processor_a', 'processor_b', 'processor_c', 'processor_d']
        )

    def processor(name, delay=0.5):
        """Individual processor with simulated work"""
        def process_func(state):
            time.sleep(delay)  # Simulate processing time
            return {
                'results': [f'{name} processed item in {delay}s']
            }
        return process_func

    def combine_results(state):
        """Fan-in: Combine all parallel results"""
        results = state.get('results', [])
        total_time = time.time() - state.get('started_at', time.time())

        return {
            'summary': f'Processed {len(results)} items in {total_time:.2f}s',
            'status': 'completed'
        }

    # Build graph
    (graph
     .add_node('start', distribute_work)
     .add_node('processor_a', processor('A', 0.3))
     .add_node('processor_b', processor('B', 0.5))
     .add_node('processor_c', processor('C', 0.2))
     .add_node('processor_d', processor('D', 0.4))
     .add_node('combiner', combine_results)
     .set_entry_point('start')

     # Fan-in: All processors → combiner
     .add_edge('processor_a', 'combiner')
     .add_edge('processor_b', 'combiner')
     .add_edge('processor_c', 'combiner')
     .add_edge('processor_d', 'combiner'))

    return graph.compile()

# Usage
app = create_basic_fanout()
result = app.invoke({'work_items': ['item1', 'item2', 'item3', 'item4']})
print(result['summary'])  # "Processed 4 items in 0.50s" (not 1.4s!)
```

**Key Benefits:**
- ✅ 4 processors run simultaneously (parallel speedup)
- ✅ Combiner waits for ALL to complete (synchronization)
- ✅ Results automatically merged with `extend` reducer

**Use Cases:**
- Batch processing multiple files
- Parallel API calls
- Independent data transformations
- Map-reduce style operations

## 🌊 Pattern 2: Pipeline with Parallel Stages

Chain multiple fan-out/fan-in stages for complex processing pipelines.

### Structure
```
    input
      |
  preprocess
   /  |  \
  A1  A2  A3
   \  |  /
  stage1_combine
   /  |  \
  B1  B2  B3
   \  |  /
 final_combine
```

### Implementation
```python
def create_pipeline():
    graph = StateGraph(state_reducers={
        'stage1_results': 'extend',
        'stage2_results': 'extend',
        'processing_log': 'extend'
    })

    def preprocess(state):
        """Initial preprocessing before parallel stages"""
        return Command(
            update={
                'processed_input': f"Cleaned: {state.get('raw_input', '')}",
                'processing_log': ['Preprocessing completed']
            },
            goto=['stage1_a', 'stage1_b', 'stage1_c']  # First fan-out
        )

    def stage1_processor(name):
        def process(state):
            # Stage 1 processing
            return {
                'stage1_results': [f'Stage1-{name}: {state["processed_input"]}'],
                'processing_log': [f'Stage 1 {name} completed']
            }
        return process

    def stage1_combiner(state):
        """Combine stage 1 results and fan-out to stage 2"""
        stage1_data = ' + '.join(state.get('stage1_results', []))

        return Command(
            update={
                'stage1_combined': stage1_data,
                'processing_log': ['Stage 1 combination completed']
            },
            goto=['stage2_x', 'stage2_y', 'stage2_z']  # Second fan-out
        )

    def stage2_processor(name):
        def process(state):
            # Stage 2 processing using stage 1 results
            return {
                'stage2_results': [f'Stage2-{name}: {state["stage1_combined"]}'],
                'processing_log': [f'Stage 2 {name} completed']
            }
        return process

    def final_combiner(state):
        """Final combination of all results"""
        return {
            'final_result': {
                'stage1_count': len(state.get('stage1_results', [])),
                'stage2_count': len(state.get('stage2_results', [])),
                'total_steps': len(state.get('processing_log', [])),
                'combined_output': ' | '.join(state.get('stage2_results', []))
            }
        }

    # Build the pipeline
    (graph
     .add_node('preprocess', preprocess)
     .add_node('stage1_a', stage1_processor('A'))
     .add_node('stage1_b', stage1_processor('B'))
     .add_node('stage1_c', stage1_processor('C'))
     .add_node('stage1_combine', stage1_combiner)
     .add_node('stage2_x', stage2_processor('X'))
     .add_node('stage2_y', stage2_processor('Y'))
     .add_node('stage2_z', stage2_processor('Z'))
     .add_node('final_combine', final_combiner)
     .set_entry_point('preprocess')

     # Stage 1 fan-in
     .add_edge('stage1_a', 'stage1_combine')
     .add_edge('stage1_b', 'stage1_combine')
     .add_edge('stage1_c', 'stage1_combine')

     # Stage 2 fan-in
     .add_edge('stage2_x', 'final_combine')
     .add_edge('stage2_y', 'final_combine')
     .add_edge('stage2_z', 'final_combine'))

    return graph.compile()

# Usage
pipeline = create_pipeline()
result = pipeline.invoke({'raw_input': 'user data'})
print(result['final_result'])
```

**Use Cases:**
- Multi-stage data processing
- ML pipelines (preprocess → feature extraction → modeling)
- Content processing (parse → analyze → generate)

## 🔀 Pattern 3: Conditional Parallel Routing

Route to different parallel processors based on input characteristics.

### Implementation
```python
def create_conditional_router():
    graph = StateGraph(state_reducers={'analysis': 'extend'})

    def content_classifier(state):
        """Analyze input and route to appropriate processors"""
        content = state.get('content', '')
        content_type = state.get('type', 'unknown')

        if content_type == 'text':
            return Command(
                update={'classification': 'text_content'},
                goto=['text_sentiment', 'text_entities', 'text_keywords']
            )
        elif content_type == 'image':
            return Command(
                update={'classification': 'image_content'},
                goto=['image_objects', 'image_faces', 'image_text']
            )
        elif content_type == 'audio':
            return Command(
                update={'classification': 'audio_content'},
                goto=['audio_transcribe', 'audio_sentiment']
            )
        else:
            return Command(
                update={'classification': 'unknown_content'},
                goto='generic_processor'
            )

    # Text processors
    def text_sentiment(state):
        return {'analysis': [{'type': 'sentiment', 'result': 'positive'}]}

    def text_entities(state):
        return {'analysis': [{'type': 'entities', 'result': ['Person', 'Location']}]}

    def text_keywords(state):
        return {'analysis': [{'type': 'keywords', 'result': ['important', 'data']}]}

    # Image processors
    def image_objects(state):
        return {'analysis': [{'type': 'objects', 'result': ['car', 'tree', 'person']}]}

    def image_faces(state):
        return {'analysis': [{'type': 'faces', 'result': 2}]}

    def image_text(state):
        return {'analysis': [{'type': 'ocr', 'result': 'Street Sign'}]}

    # Audio processors
    def audio_transcribe(state):
        return {'analysis': [{'type': 'transcript', 'result': 'Hello world'}]}

    def audio_sentiment(state):
        return {'analysis': [{'type': 'audio_sentiment', 'result': 'neutral'}]}

    # Generic fallback
    def generic_processor(state):
        return {'analysis': [{'type': 'generic', 'result': 'processed'}]}

    def results_aggregator(state):
        """Combine results from whichever path was taken"""
        analyses = state.get('analysis', [])
        classification = state.get('classification', 'unknown')

        return {
            'final_analysis': {
                'content_type': classification,
                'analyses_performed': len(analyses),
                'results': {item['type']: item['result'] for item in analyses}
            }
        }

    # Build graph with all possible paths
    (graph
     .add_node('classifier', content_classifier)
     .add_node('text_sentiment', text_sentiment)
     .add_node('text_entities', text_entities)
     .add_node('text_keywords', text_keywords)
     .add_node('image_objects', image_objects)
     .add_node('image_faces', image_faces)
     .add_node('image_text', image_text)
     .add_node('audio_transcribe', audio_transcribe)
     .add_node('audio_sentiment', audio_sentiment)
     .add_node('generic_processor', generic_processor)
     .add_node('aggregator', results_aggregator)
     .set_entry_point('classifier')

     # All paths lead to aggregator
     .add_edge('text_sentiment', 'aggregator')
     .add_edge('text_entities', 'aggregator')
     .add_edge('text_keywords', 'aggregator')
     .add_edge('image_objects', 'aggregator')
     .add_edge('image_faces', 'aggregator')
     .add_edge('image_text', 'aggregator')
     .add_edge('audio_transcribe', 'aggregator')
     .add_edge('audio_sentiment', 'aggregator')
     .add_edge('generic_processor', 'aggregator'))

    return graph.compile()

# Test different content types
router = create_conditional_router()

# Text processing
result = router.invoke({'type': 'text', 'content': 'Great product review!'})
print(f"Text: {result['final_analysis']}")

# Image processing  
result = router.invoke({'type': 'image', 'content': 'photo.jpg'})
print(f"Image: {result['final_analysis']}")
```

**Use Cases:**
- Multi-modal content analysis
- Request routing based on user type
- Dynamic workflow selection
- A/B testing with parallel variants

## 🔄 Pattern 4: Map-Reduce with Dynamic Parallelism

Process collections with automatic parallelization based on collection size.

### Implementation
```python
def create_map_reduce():
    graph = StateGraph(state_reducers={
        'mapped_results': 'extend',
        'processing_stats': 'merge'
    })

    def map_phase(state):
        """Dynamically create parallel mappers based on input size"""
        items = state.get('items', [])

        if len(items) <= 3:
            # Small collection - process sequentially
            return Command(
                update={'map_strategy': 'sequential'},
                goto='single_mapper'
            )
        else:
            # Large collection - parallel processing
            # Split into chunks for parallel processing
            chunk_size = max(1, len(items) // 4)  # Aim for 4 parallel processors

            return Command(
                update={
                    'map_strategy': 'parallel',
                    'chunk_size': chunk_size,
                    'total_items': len(items)
                },
                goto=['mapper_1', 'mapper_2', 'mapper_3', 'mapper_4']
            )

    def single_mapper(state):
        """Process all items sequentially"""
        items = state.get('items', [])
        results = [f"Processed: {item}" for item in items]

        return {
            'mapped_results': results,
            'processing_stats': {'mode': 'sequential', 'items': len(items)}
        }

    def parallel_mapper(mapper_id):
        """Process a chunk of items in parallel"""
        def mapper_func(state):
            items = state.get('items', [])
            chunk_size = state.get('chunk_size', 1)
            start_idx = (mapper_id - 1) * chunk_size
            end_idx = start_idx + chunk_size

            # Process this mapper's chunk
            chunk = items[start_idx:end_idx] if start_idx < len(items) else []
            results = [f"Mapper{mapper_id} processed: {item}" for item in chunk]

            return {
                'mapped_results': results,
                'processing_stats': {f'mapper_{mapper_id}_items': len(chunk)}
            }
        return mapper_func

    def reduce_phase(state):
        """Combine all mapped results"""
        results = state.get('mapped_results', [])
        stats = state.get('processing_stats', {})
        strategy = state.get('map_strategy', 'unknown')

        return {
            'final_result': {
                'strategy': strategy,
                'total_processed': len(results),
                'results': results[:5],  # Show first 5 results
                'stats': stats
            }
        }

    # Build graph
    (graph
     .add_node('map_phase', map_phase)
     .add_node('single_mapper', single_mapper)
     .add_node('mapper_1', parallel_mapper(1))
     .add_node('mapper_2', parallel_mapper(2))
     .add_node('mapper_3', parallel_mapper(3))
     .add_node('mapper_4', parallel_mapper(4))
     .add_node('reduce_phase', reduce_phase)
     .set_entry_point('map_phase')

     # Both sequential and parallel paths lead to reduce
     .add_edge('single_mapper', 'reduce_phase')
     .add_edge('mapper_1', 'reduce_phase')
     .add_edge('mapper_2', 'reduce_phase')
     .add_edge('mapper_3', 'reduce_phase')
     .add_edge('mapper_4', 'reduce_phase'))

    return graph.compile()

# Test with different collection sizes
map_reduce = create_map_reduce()

# Small collection (sequential)
small_result = map_reduce.invoke({'items': ['a', 'b']})
print(f"Small: {small_result['final_result']['strategy']}")

# Large collection (parallel)
large_items = [f"item_{i}" for i in range(20)]
large_result = map_reduce.invoke({'items': large_items})
print(f"Large: {large_result['final_result']['strategy']}")
```

**Use Cases:**
- Processing large datasets
- Batch operations with automatic scaling
- File processing with size-based optimization
- API batch requests

## 🔁 Pattern 5: Retry with Parallel Fallbacks

Handle failures by retrying or using alternative parallel processors.

### Implementation
```python
def create_resilient_processor():
    graph = StateGraph(state_reducers={
        'attempts': 'extend',
        'results': 'extend'
    })

    def primary_processor(state):
        """Main processor that might fail"""
        failure_rate = state.get('failure_rate', 0.3)

        if random.random() < failure_rate:
            # Simulate failure
            return Command(
                update={
                    'attempts': ['primary_failed'],
                    'error': 'Primary processor failed'
                },
                goto=['backup_a', 'backup_b', 'backup_c']  # Fan-out to backups
            )
        else:
            # Success
            return Command(
                update={
                    'attempts': ['primary_success'],
                    'results': ['Primary processing completed']
                },
                goto='success_handler'
            )

    def backup_processor(name, reliability=0.8):
        """Backup processor with different reliability"""
        def backup_func(state):
            if random.random() < reliability:
                return {
                    'attempts': [f'{name}_success'],
                    'results': [f'{name} backup processing completed']
                }
            else:
                return {
                    'attempts': [f'{name}_failed'],
                    'error': f'{name} backup failed'
                }
        return backup_func

    def success_handler(state):
        """Handle successful processing"""
        return {'status': 'completed', 'method': 'primary'}

    def backup_combiner(state):
        """Combine backup results - succeed if any backup works"""
        results = state.get('results', [])
        attempts = state.get('attempts', [])

        if results:
            # At least one backup succeeded
            successful_attempts = [a for a in attempts if 'success' in a]
            return {
                'status': 'completed',
                'method': 'backup',
                'successful_backups': successful_attempts,
                'final_result': results[0]  # Use first successful result
            }
        else:
            # All backups failed
            return {
                'status': 'failed',
                'method': 'all_failed',
                'all_attempts': attempts
            }

    # Build resilient graph
    (graph
     .add_node('primary', primary_processor)
     .add_node('backup_a', backup_processor('BackupA', 0.9))
     .add_node('backup_b', backup_processor('BackupB', 0.8))
     .add_node('backup_c', backup_processor('BackupC', 0.7))
     .add_node('success_handler', success_handler)
     .add_node('backup_combiner', backup_combiner)
     .set_entry_point('primary')

     # Success path
     .add_edge('success_handler', 'backup_combiner')

     # Backup paths all lead to combiner
     .add_edge('backup_a', 'backup_combiner')
     .add_edge('backup_b', 'backup_combiner')
     .add_edge('backup_c', 'backup_combiner'))

    return graph.compile()

# Test resilience
processor = create_resilient_processor()

for i in range(5):
    result = processor.invoke({'failure_rate': 0.5})
    print(f"Run {i+1}: {result['status']} via {result['method']}")
```

**Use Cases:**
- API calls with multiple providers
- File processing with fallback methods
- Service redundancy and failover
- Multi-cloud deployment strategies

## 🎯 Pattern Best Practices

### 1. **State Design**
```python
# Good: Use appropriate reducers for your data types
graph = StateGraph(state_reducers={
    'results': 'extend',      # For collecting parallel results
    'metadata': 'merge',      # For combining metadata
    'latest_status': 'set',   # For status updates
    'error_log': 'extend'     # For collecting errors
})
```

### 2. **Error Handling**
```python
def robust_processor(state):
    try:
        result = do_processing(state)
        return {'results': [result]}
    except Exception as e:
        return {
            'errors': [{'node': 'processor', 'error': str(e)}],
            'status': 'failed'
        }
```

### 3. **Performance Monitoring**
```python
def timed_processor(state):
    start_time = time.time()
    result = do_work(state)

    return {
        'results': [result],
        'timing': [{
            'node': 'processor',
            'duration': time.time() - start_time
        }]
    }
```

### 4. **Resource Management**
```python
# Configure concurrency limits for resource-intensive operations
app = graph.compile(
    use_parallel_engine=True,
    max_concurrent=3  # Limit to 3 parallel nodes
)
```

## 🚀 Combining Patterns

Real-world applications often combine multiple patterns:

```python
def create_advanced_workflow():
    """Combines conditional routing + map-reduce + resilience"""
    graph = StateGraph(state_reducers={
        'processed_items': 'extend',
        'errors': 'extend',
        'metrics': 'merge'
    })

    # Pattern 1: Conditional routing
    def input_classifier(state):
        if state.get('batch_size', 0) > 100:
            return Command(goto='large_batch_handler')
        else:
            return Command(goto='small_batch_handler')

    # Pattern 2: Map-reduce for large batches
    def large_batch_handler(state):
        return Command(goto=['mapper_1', 'mapper_2', 'mapper_3'])

    # Pattern 3: Simple processing for small batches
    def small_batch_handler(state):
        return Command(goto='simple_processor')

    # Pattern 4: Resilient processing with backups
    def simple_processor(state):
        if random.random() < 0.2:  # 20% failure rate
            return Command(goto=['backup_processor'])
        return {'processed_items': ['simple_result']}

    # Build combined workflow...
    # (implementation details omitted for brevity)
```

## 🎓 Next Steps

Now that you understand the core patterns:

1. **Practice**: Implement each pattern with your own data
2. **Combine**: Mix patterns to solve complex problems  
3. **Optimize**: Use performance monitoring to tune your graphs
4. **Scale**: Apply these patterns to real production workloads

**Master these patterns, and you'll be able to build sophisticated parallel AI agent workflows!** 🚀