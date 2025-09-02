# Building Workflows: From Simple Chains to Complex Parallel Graphs

*Learn how to construct sophisticated AI agent workflows using GraphFlow's powerful parallel execution engine*

## 🎯 What You'll Learn

This guide takes you from basic workflow concepts to advanced parallel agent architectures. You'll master:

1. **Linear workflows** - Simple sequential processing
2. **Parallel workflows** - Concurrent execution patterns
3. **Complex graphs** - Multi-path decision workflows
4. **AI agent patterns** - LLM-powered intelligent workflows
5. **Production patterns** - Robust, scalable workflow design

## 🚀 Quick Start: Your First Parallel Workflow

Let's start with a simple example that shows the power of parallel execution:

```python
from graphflow import StateGraph, Command

def create_research_workflow():
    """A research agent that gathers information from multiple sources in parallel"""

    # Configure parallel execution with state merging
    graph = StateGraph(state_reducers={
        'research_results': 'extend',    # Collect all research findings
        'source_metadata': 'merge',      # Combine metadata from all sources
        'processing_log': 'extend'       # Track processing steps
    })

    def analyze_query(state):
        """Analyze the research query and determine information needs"""
        query = state.get('query', '')

        # Determine which research sources to use
        return Command(
            update={
                'analyzed_query': query,
                'research_plan': 'multi-source parallel research',
                'processing_log': [f'Analyzed query: {query}']
            },
            goto=['web_search', 'academic_search', 'news_search', 'social_search']
        )

    def web_search(state):
        """Simulate web search results"""
        import time
        time.sleep(0.5)  # Simulate API call

        return {
            'research_results': [{
                'source': 'web_search',
                'findings': 'Found 5 relevant web articles',
                'confidence': 0.8
            }],
            'source_metadata': {
                'web_search': {'articles_found': 5, 'duration': 0.5}
            },
            'processing_log': ['Web search completed']
        }

    def academic_search(state):
        """Simulate academic database search"""
        import time
        time.sleep(0.7)  # Simulate slower academic search

        return {
            'research_results': [{
                'source': 'academic_search',
                'findings': 'Found 3 peer-reviewed papers',
                'confidence': 0.95
            }],
            'source_metadata': {
                'academic_search': {'papers_found': 3, 'duration': 0.7}
            },
            'processing_log': ['Academic search completed']
        }

    def news_search(state):
        """Simulate news search"""
        import time
        time.sleep(0.3)  # Fast news API

        return {
            'research_results': [{
                'source': 'news_search',
                'findings': 'Found 8 recent news articles',
                'confidence': 0.7
            }],
            'source_metadata': {
                'news_search': {'articles_found': 8, 'duration': 0.3}
            },
            'processing_log': ['News search completed']
        }

    def social_search(state):
        """Simulate social media search"""
        import time
        time.sleep(0.4)

        return {
            'research_results': [{
                'source': 'social_search',
                'findings': 'Found 12 relevant social posts',
                'confidence': 0.6
            }],
            'source_metadata': {
                'social_search': {'posts_found': 12, 'duration': 0.4}
            },
            'processing_log': ['Social search completed']
        }

    def synthesize_research(state):
        """Combine all research findings into a comprehensive report"""
        results = state.get('research_results', [])
        metadata = state.get('source_metadata', {})

        # Calculate total research quality
        total_confidence = sum(r.get('confidence', 0) for r in results) / len(results)

        return {
            'final_report': {
                'summary': f'Research completed using {len(results)} sources',
                'sources_used': [r['source'] for r in results],
                'average_confidence': total_confidence,
                'detailed_findings': results,
                'source_statistics': metadata
            },
            'processing_log': ['Research synthesis completed']
        }

    # Build the graph
    (graph
     .add_node('analyzer', analyze_query)
     .add_node('web_search', web_search)
     .add_node('academic_search', academic_search)
     .add_node('news_search', news_search)
     .add_node('social_search', social_search)
     .add_node('synthesizer', synthesize_research)
     .set_entry_point('analyzer')

     # All search nodes feed into synthesizer (fan-in pattern)
     .add_edge('web_search', 'synthesizer')
     .add_edge('academic_search', 'synthesizer')
     .add_edge('news_search', 'synthesizer')
     .add_edge('social_search', 'synthesizer'))

    return graph.compile()

# Test the research workflow
research_agent = create_research_workflow()
result = research_agent.invoke({'query': 'climate change impacts on agriculture'})

print("🔍 Research Results:")
print(f"Sources: {result['final_report']['sources_used']}")
print(f"Confidence: {result['final_report']['average_confidence']:.2f}")
print(f"Steps: {len(result['processing_log'])}")
```

**What makes this powerful:**
- ✅ **4 searches run in parallel** (not sequentially)
- ✅ **Results automatically merged** using reducers
- ✅ **Synchronization point** - synthesizer waits for all searches
- ✅ **Execution time = slowest search** (0.7s, not 1.9s total)

## 🏗️ Workflow Architecture Patterns

### Pattern 1: Linear Pipeline
**When to use:** Sequential processing where each step depends on the previous

```python
def create_linear_pipeline():
    """Simple sequential processing"""
    graph = StateGraph()

    def step1(state):
        return {'result': 'step1_output', 'next': 'step2'}

    def step2(state):
        previous = state.get('result', '')
        return {'result': f'{previous} -> step2_output', 'next': 'step3'}

    def step3(state):
        previous = state.get('result', '')
        return {'result': f'{previous} -> step3_output'}

    # Linear chain: step1 → step2 → step3
    (graph
     .add_node('step1', step1)
     .add_node('step2', step2)
     .add_node('step3', step3)
     .set_entry_point('step1')
     .add_edge('step1', 'step2')
     .add_edge('step2', 'step3'))

    return graph.compile()
```

### Pattern 2: Fan-Out/Fan-In
**When to use:** Parallel processing with result aggregation

```python
def create_fanout_pipeline():
    """Distribute work, process in parallel, combine results"""
    graph = StateGraph(state_reducers={'results': 'extend'})

    def distribute(state):
        work = state.get('work_items', [])
        return Command(
            update={'distributed_at': time.time()},
            goto=['worker1', 'worker2', 'worker3']  # Fan-out
        )

    def worker(name):
        def work_func(state):
            return {'results': [f'{name}_completed']}
        return work_func

    def combine(state):
        results = state.get('results', [])
        return {'summary': f'Combined {len(results)} results'}

    # Fan-out/fan-in pattern
    (graph
     .add_node('distribute', distribute)
     .add_node('worker1', worker('Worker1'))
     .add_node('worker2', worker('Worker2'))
     .add_node('worker3', worker('Worker3'))
     .add_node('combine', combine)
     .set_entry_point('distribute')

     # Fan-in: all workers → combine
     .add_edge('worker1', 'combine')
     .add_edge('worker2', 'combine')
     .add_edge('worker3', 'combine'))

    return graph.compile()
```

### Pattern 3: Conditional Branching
**When to use:** Different paths based on input or conditions

```python
def create_conditional_workflow():
    """Route to different processors based on conditions"""
    graph = StateGraph(state_reducers={'analysis': 'extend'})

    def classifier(state):
        """Route based on input type"""
        input_type = state.get('type', 'unknown')

        if input_type == 'urgent':
            return Command(goto='priority_processor')
        elif input_type == 'complex':
            return Command(goto=['detailed_analyzer', 'expert_reviewer'])
        else:
            return Command(goto='standard_processor')

    def priority_processor(state):
        return {'analysis': [{'type': 'priority', 'result': 'Fast processing'}]}

    def detailed_analyzer(state):
        return {'analysis': [{'type': 'detailed', 'result': 'Deep analysis'}]}

    def expert_reviewer(state):
        return {'analysis': [{'type': 'expert', 'result': 'Expert review'}]}

    def standard_processor(state):
        return {'analysis': [{'type': 'standard', 'result': 'Standard processing'}]}

    def final_processor(state):
        analyses = state.get('analysis', [])
        return {'final_result': f'Processed with {len(analyses)} analyses'}

    # Conditional branching with convergence
    (graph
     .add_node('classifier', classifier)
     .add_node('priority_processor', priority_processor)
     .add_node('detailed_analyzer', detailed_analyzer)
     .add_node('expert_reviewer', expert_reviewer)
     .add_node('standard_processor', standard_processor)
     .add_node('final_processor', final_processor)
     .set_entry_point('classifier')

     # All paths converge at final_processor
     .add_edge('priority_processor', 'final_processor')
     .add_edge('detailed_analyzer', 'final_processor')
     .add_edge('expert_reviewer', 'final_processor')
     .add_edge('standard_processor', 'final_processor'))

    return graph.compile()
```

### Pattern 4: Multi-Stage Pipeline
**When to use:** Complex processing with multiple parallel stages

```python
def create_multistage_pipeline():
    """Multiple stages of parallel processing"""
    graph = StateGraph(state_reducers={
        'stage1_results': 'extend',
        'stage2_results': 'extend',
        'final_analysis': 'merge'
    })

    def input_processor(state):
        """Prepare input for first stage"""
        return Command(
            update={'processed_input': 'preprocessed_data'},
            goto=['stage1_a', 'stage1_b']  # First parallel stage
        )

    def stage1_processor(name):
        def processor(state):
            return {'stage1_results': [f'Stage1_{name}_result']}
        return processor

    def stage1_combiner(state):
        """Combine stage 1 and launch stage 2"""
        stage1_data = state.get('stage1_results', [])
        return Command(
            update={'stage1_summary': f'Stage1 completed with {len(stage1_data)} results'},
            goto=['stage2_x', 'stage2_y', 'stage2_z']  # Second parallel stage
        )

    def stage2_processor(name):
        def processor(state):
            return {'stage2_results': [f'Stage2_{name}_result']}
        return processor

    def final_combiner(state):
        """Final combination of all results"""
        stage1 = state.get('stage1_results', [])
        stage2 = state.get('stage2_results', [])

        return {
            'final_analysis': {
                'stage1_count': len(stage1),
                'stage2_count': len(stage2),
                'total_processing_stages': 2
            }
        }

    # Multi-stage pipeline
    (graph
     .add_node('input_processor', input_processor)
     .add_node('stage1_a', stage1_processor('A'))
     .add_node('stage1_b', stage1_processor('B'))
     .add_node('stage1_combiner', stage1_combiner)
     .add_node('stage2_x', stage2_processor('X'))
     .add_node('stage2_y', stage2_processor('Y'))
     .add_node('stage2_z', stage2_processor('Z'))
     .add_node('final_combiner', final_combiner)
     .set_entry_point('input_processor')

     # Stage 1 convergence
     .add_edge('stage1_a', 'stage1_combiner')
     .add_edge('stage1_b', 'stage1_combiner')

     # Stage 2 convergence
     .add_edge('stage2_x', 'final_combiner')
     .add_edge('stage2_y', 'final_combiner')
     .add_edge('stage2_z', 'final_combiner'))

    return graph.compile()
```

## 🤖 AI Agent Workflow Patterns

### Pattern 1: Multi-Expert Analysis
**Use case:** Get multiple AI perspectives on complex problems

```python
def create_multi_expert_agent():
    """Multiple AI experts analyze the same problem in parallel"""
    graph = StateGraph(state_reducers={
        'expert_opinions': 'extend',
        'confidence_scores': 'extend'
    })

    def problem_distributor(state):
        """Send problem to all expert agents"""
        problem = state.get('problem', '')
        return Command(
            update={'distributed_problem': problem},
            goto=['technical_expert', 'business_expert', 'creative_expert']
        )

    def technical_expert(state):
        """AI expert focused on technical analysis"""
        problem = state.get('problem', '')
        # In real implementation, this would call an LLM with technical prompts
        analysis = f"Technical analysis of: {problem}"

        return {
            'expert_opinions': [{
                'expert': 'technical',
                'analysis': analysis,
                'focus': 'feasibility and implementation'
            }],
            'confidence_scores': [0.85]
        }

    def business_expert(state):
        """AI expert focused on business implications"""
        problem = state.get('problem', '')
        analysis = f"Business analysis of: {problem}"

        return {
            'expert_opinions': [{
                'expert': 'business',
                'analysis': analysis,
                'focus': 'market impact and ROI'
            }],
            'confidence_scores': [0.78]
        }

    def creative_expert(state):
        """AI expert focused on creative solutions"""
        problem = state.get('problem', '')
        analysis = f"Creative analysis of: {problem}"

        return {
            'expert_opinions': [{
                'expert': 'creative',
                'analysis': analysis,
                'focus': 'innovative approaches'
            }],
            'confidence_scores': [0.72]
        }

    def synthesize_experts(state):
        """Combine all expert opinions into final recommendation"""
        opinions = state.get('expert_opinions', [])
        confidence_scores = state.get('confidence_scores', [])

        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        return {
            'final_recommendation': {
                'expert_count': len(opinions),
                'average_confidence': avg_confidence,
                'synthesis': 'Combined expert analysis',
                'detailed_opinions': opinions
            }
        }

    # Build multi-expert workflow
    (graph
     .add_node('distributor', problem_distributor)
     .add_node('technical_expert', technical_expert)
     .add_node('business_expert', business_expert)
     .add_node('creative_expert', creative_expert)
     .add_node('synthesizer', synthesize_experts)
     .set_entry_point('distributor')

     .add_edge('technical_expert', 'synthesizer')
     .add_edge('business_expert', 'synthesizer')
     .add_edge('creative_expert', 'synthesizer'))

    return graph.compile()
```

### Pattern 2: Iterative Refinement Agent
**Use case:** Improve results through multiple parallel refinement passes

```python
def create_iterative_refinement_agent():
    """Iteratively refine results using parallel processors"""
    graph = StateGraph(state_reducers={
        'refinement_history': 'extend',
        'current_quality': 'set',
        'improvements': 'extend'
    })

    def quality_checker(state):
        """Check if current result meets quality threshold"""
        current_result = state.get('current_result', '')
        iteration = state.get('iteration', 0)
        quality = len(current_result) * 10  # Simple quality metric

        if quality >= 500 or iteration >= 3:  # Good enough or max iterations
            return Command(
                update={'final_quality': quality, 'status': 'completed'},
                goto='final_processor'
            )
        else:
            return Command(
                update={
                    'current_quality': quality,
                    'iteration': iteration + 1,
                    'status': 'refining'
                },
                goto=['grammar_refiner', 'style_refiner', 'content_refiner']
            )

    def grammar_refiner(state):
        """Refine grammar and structure"""
        current = state.get('current_result', '')
        refined = f"{current} [grammar-refined]"

        return {
            'refinement_history': ['Grammar refinement applied'],
            'improvements': [{'type': 'grammar', 'improvement': 'Fixed grammar issues'}]
        }

    def style_refiner(state):
        """Refine writing style"""
        current = state.get('current_result', '')
        refined = f"{current} [style-refined]"

        return {
            'refinement_history': ['Style refinement applied'],
            'improvements': [{'type': 'style', 'improvement': 'Improved writing style'}]
        }

    def content_refiner(state):
        """Refine content depth"""
        current = state.get('current_result', '')
        refined = f"{current} [content-refined]"

        return {
            'refinement_history': ['Content refinement applied'],
            'improvements': [{'type': 'content', 'improvement': 'Enhanced content depth'}],
            'current_result': refined  # Update the result being refined
        }

    def refinement_combiner(state):
        """Combine refinements and check quality again"""
        improvements = state.get('improvements', [])
        current = state.get('current_result', '')

        # Apply all improvements (simplified)
        refined_result = current + f" [refined-{len(improvements)}-ways]"

        return Command(
            update={
                'current_result': refined_result,
                'refinement_history': [f'Applied {len(improvements)} refinements']
            },
            goto='quality_checker'  # Loop back to quality check
        )

    def final_processor(state):
        """Final processing when quality is sufficient"""
        history = state.get('refinement_history', [])
        quality = state.get('final_quality', 0)

        return {
            'final_output': {
                'result': state.get('current_result', ''),
                'quality_score': quality,
                'refinement_passes': len(history),
                'total_improvements': len(state.get('improvements', []))
            }
        }

    # Build iterative refinement workflow
    (graph
     .add_node('quality_checker', quality_checker)
     .add_node('grammar_refiner', grammar_refiner)
     .add_node('style_refiner', style_refiner)
     .add_node('content_refiner', content_refiner)
     .add_node('refinement_combiner', refinement_combiner)
     .add_node('final_processor', final_processor)
     .set_entry_point('quality_checker')

     # Refiners feed into combiner
     .add_edge('grammar_refiner', 'refinement_combiner')
     .add_edge('style_refiner', 'refinement_combiner')
     .add_edge('content_refiner', 'refinement_combiner'))

    return graph.compile()
```

## 🏭 Production Workflow Patterns

### Pattern 1: Error Handling and Retries
**Essential for:** Robust production workflows

```python
def create_resilient_workflow():
    """Production workflow with error handling and retries"""
    graph = StateGraph(state_reducers={
        'attempts': 'extend',
        'errors': 'extend',
        'results': 'extend'
    })

    def robust_processor(state):
        """Main processor with error handling"""
        try:
            # Simulate processing that might fail
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("Processing failed")

            return {
                'results': ['Primary processing succeeded'],
                'attempts': ['primary_success']
            }
        except Exception as e:
            return Command(
                update={
                    'errors': [{'source': 'primary', 'error': str(e)}],
                    'attempts': ['primary_failed']
                },
                goto=['backup_processor_1', 'backup_processor_2']  # Try backups
            )

    def backup_processor(name, success_rate=0.8):
        """Backup processor with different reliability"""
        def processor(state):
            import random
            if random.random() < success_rate:
                return {
                    'results': [f'{name} backup processing succeeded'],
                    'attempts': [f'{name}_success']
                }
            else:
                return {
                    'errors': [{'source': name, 'error': f'{name} backup failed'}],
                    'attempts': [f'{name}_failed']
                }
        return processor

    def error_handler(state):
        """Handle final results or errors"""
        results = state.get('results', [])
        errors = state.get('errors', [])
        attempts = state.get('attempts', [])

        if results:
            return {
                'status': 'success',
                'final_result': results[0],
                'fallback_used': 'primary' not in attempts[0] if attempts else False
            }
        else:
            return {
                'status': 'failed',
                'error_summary': f'{len(errors)} errors occurred',
                'all_attempts': attempts
            }

    # Build resilient workflow
    (graph
     .add_node('primary', robust_processor)
     .add_node('backup_1', backup_processor('Backup1', 0.9))
     .add_node('backup_2', backup_processor('Backup2', 0.8))
     .add_node('error_handler', error_handler)
     .set_entry_point('primary')

     # All paths lead to error handler
     .add_edge('primary', 'error_handler')
     .add_edge('backup_1', 'error_handler')
     .add_edge('backup_2', 'error_handler'))

    return graph.compile()
```

### Pattern 2: Performance Monitoring
**Essential for:** Production optimization

```python
def create_monitored_workflow():
    """Workflow with comprehensive performance monitoring"""
    graph = StateGraph(state_reducers={
        'performance_metrics': 'extend',
        'resource_usage': 'merge',
        'processing_events': 'extend'
    })

    def monitored_processor(node_name):
        """Wrap processors with performance monitoring"""
        def processor(state):
            import time
            import psutil

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Simulate processing
            time.sleep(0.1)
            result = f'{node_name} processing completed'

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            return {
                'performance_metrics': [{
                    'node': node_name,
                    'duration': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'timestamp': end_time
                }],
                'processing_events': [f'{node_name} completed at {end_time}'],
                f'{node_name}_result': result
            }
        return processor

    def performance_analyzer(state):
        """Analyze performance metrics"""
        metrics = state.get('performance_metrics', [])

        if not metrics:
            return {'performance_summary': 'No metrics available'}

        total_duration = sum(m['duration'] for m in metrics)
        avg_duration = total_duration / len(metrics)
        max_duration = max(m['duration'] for m in metrics)
        total_memory = sum(m.get('memory_delta', 0) for m in metrics)

        return {
            'performance_summary': {
                'total_nodes': len(metrics),
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'max_duration': max_duration,
                'total_memory_usage': total_memory,
                'detailed_metrics': metrics
            }
        }

    # Build monitored workflow
    (graph
     .add_node('processor_1', monitored_processor('Processor1'))
     .add_node('processor_2', monitored_processor('Processor2'))
     .add_node('processor_3', monitored_processor('Processor3'))
     .add_node('analyzer', performance_analyzer)
     .set_entry_point('processor_1')

     .add_edge('processor_1', 'processor_2')
     .add_edge('processor_2', 'processor_3')
     .add_edge('processor_3', 'analyzer'))

    return graph.compile()
```

## 🎯 Best Practices for Workflow Design

### 1. **Start Simple, Add Complexity Gradually**
```python
# ✅ Good: Start with simple linear workflow
def v1_workflow():
    # Linear: input → process → output
    pass

# ✅ Good: Add parallelism where it makes sense
def v2_workflow():
    # Parallel: input → [process_a, process_b] → combine → output
    pass

# ✅ Good: Add sophisticated patterns as needed
def v3_workflow():
    # Complex: conditional routing + parallel processing + error handling
    pass
```

### 2. **Design for Your Data Flow**
```python
# ✅ Good: Match workflow structure to data dependencies
# If A and B can run independently, make them parallel
# If C needs both A and B, make it a fan-in point

graph = StateGraph()
# A and B are independent
.add_node('process_a', process_a)
.add_node('process_b', process_b)
# C needs both A and B
.add_node('combine_c', combine_c)
.add_edge('process_a', 'combine_c')
.add_edge('process_b', 'combine_c')
```

### 3. **Use Appropriate State Reducers**
```python
# ✅ Good: Choose reducers that match your data patterns
graph = StateGraph(state_reducers={
    'results': 'extend',        # Collecting parallel results
    'metadata': 'merge',        # Building complex objects
    'final_status': 'set',      # Single authoritative value
    'error_log': 'extend'       # Accumulating errors
})
```

### 4. **Handle Errors Gracefully**
```python
# ✅ Good: Plan for failures
def robust_node(state):
    try:
        result = risky_operation(state)
        return {'result': result}
    except Exception as e:
        return Command(
            update={'errors': [str(e)]},
            goto='error_recovery_node'
        )
```

### 5. **Monitor Performance**
```python
# ✅ Good: Include timing and resource monitoring
def timed_processor(state):
    start_time = time.time()
    result = do_processing(state)
    duration = time.time() - start_time

    return {
        'result': result,
        'performance': {'duration': duration, 'node': 'processor'}
    }
```

## 🚀 Next Steps

You now have the knowledge to build sophisticated workflows! Here's your progression path:

1. **Start with simple patterns** - Master linear and fan-out/fan-in
2. **Add complexity gradually** - Introduce conditional routing and multi-stage processing  
3. **Build AI agent workflows** - Use these patterns for LLM-powered systems
4. **Optimize for production** - Add error handling, monitoring, and resilience
5. **Create your own patterns** - Combine and adapt these patterns for your specific needs

**Remember: The best workflow is the simplest one that solves your problem effectively!** 🎯