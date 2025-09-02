# State Management: Understanding Reducers and Parallel Merging

*Master state management in parallel workflows with GraphFlow's powerful reducer system*

## 🎯 What is State Management?

In parallel graph execution, multiple nodes run simultaneously and may update the same state fields. **State management** is how we safely combine these concurrent updates without conflicts or data loss.

Think of it like multiple people editing the same document - you need rules for merging their changes!

## 🧠 Core Concepts

### The Problem: Concurrent State Updates

```python
# ❌ This creates a problem in parallel execution:
# Node A updates: {'results': ['A_result']}
# Node B updates: {'results': ['B_result']}  
# Which one wins? We might lose data!

# ✅ Solution: Use reducers to merge updates safely
graph = StateGraph(state_reducers={'results': 'extend'})
# Now both results are preserved: {'results': ['A_result', 'B_result']}
```

### What are Reducers?

**Reducers** are merge strategies that define how to combine multiple updates to the same state field. They're the "rules" for resolving conflicts when parallel nodes update the same data.

## 🔧 Built-in Reducer Types

### 1. `'extend'` - Combine Lists
**Use when:** Collecting results from parallel processors

```python
graph = StateGraph(state_reducers={'results': 'extend'})

# Node A returns: {'results': ['result_A']}
# Node B returns: {'results': ['result_B']} 
# Node C returns: {'results': ['result_C']}

# Final state: {'results': ['result_A', 'result_B', 'result_C']}
```

**Perfect for:**
- Collecting parallel processing results
- Building lists of completed tasks
- Accumulating log entries

### 2. `'append'` - Add Single Values
**Use when:** Each node contributes one item

```python
graph = StateGraph(state_reducers={'items': 'append'})

# Node A returns: {'items': 'item_A'}
# Node B returns: {'items': 'item_B'}

# Final state: {'items': ['item_A', 'item_B']}
```

**Perfect for:**
- Single value contributions
- Building collections from individual results
- Event logging

### 3. `'merge'` - Combine Dictionaries
**Use when:** Each node updates different parts of a complex object

```python
graph = StateGraph(state_reducers={'analysis': 'merge'})

# Node A returns: {'analysis': {'sentiment': 'positive'}}
# Node B returns: {'analysis': {'entities': ['Person', 'Place']}}
# Node C returns: {'analysis': {'keywords': ['important', 'data']}}

# Final state: {
#   'analysis': {
#     'sentiment': 'positive',
#     'entities': ['Person', 'Place'], 
#     'keywords': ['important', 'data']
#   }
# }
```

**Perfect for:**
- Complex analysis results
- Configuration objects
- Metadata collection

### 4. `'set'` - Last Write Wins
**Use when:** Only the final value matters

```python
graph = StateGraph(state_reducers={'status': 'set'})

# Node A returns: {'status': 'processing'}
# Node B returns: {'status': 'analyzing'}
# Node C returns: {'status': 'completed'}

# Final state: {'status': 'completed'}  # Last update wins
```

**Perfect for:**
- Status updates
- Configuration overrides
- Final result selection

## 🛠️ Custom Reducers

For complex merging logic, create custom reducer functions:

```python
def smart_counter_reducer(current_value, new_value):
    """Custom reducer that intelligently combines counters"""
    if isinstance(current_value, dict) and isinstance(new_value, dict):
        # Merge dictionaries by adding counter values
        result = current_value.copy()
        for key, value in new_value.items():
            result[key] = result.get(key, 0) + value
        return result
    else:
        # Fallback to simple addition
        return (current_value or 0) + (new_value or 0)

def priority_selector_reducer(current_value, new_value):
    """Custom reducer that keeps highest priority item"""
    if not current_value:
        return new_value
    if not new_value:
        return current_value

    # Compare priorities
    current_priority = current_value.get('priority', 0)
    new_priority = new_value.get('priority', 0)

    return new_value if new_priority > current_priority else current_value

# Use custom reducers
graph = StateGraph(state_reducers={
    'counters': smart_counter_reducer,
    'best_option': priority_selector_reducer,
    'results': 'extend'  # Mix custom and built-in reducers
})
```

## 📊 Real-World Example: Multi-Analysis Pipeline

Let's see how reducers work in a realistic scenario:

```python
from graphflow import StateGraph, Command
import time

def create_content_analyzer():
    """Analyze content with multiple parallel processors"""

    graph = StateGraph(state_reducers={
        'analysis_results': 'extend',      # Collect all analysis results
        'metadata': 'merge',               # Combine metadata from all analyzers
        'performance_stats': 'merge',      # Combine timing data
        'final_score': 'set',              # Keep the final computed score
        'processing_log': 'extend'         # Collect log entries
    })

    def distribute_content(state):
        """Send content to all analyzers"""
        content = state.get('content', '')

        return Command(
            update={
                'start_time': time.time(),
                'processing_log': [f'Starting analysis of {len(content)} chars']
            },
            goto=['sentiment_analyzer', 'entity_extractor', 'keyword_analyzer', 'quality_scorer']
        )

    def sentiment_analyzer(state):
        """Analyze sentiment with timing"""
        start = time.time()
        content = state.get('content', '')

        # Simulate analysis
        time.sleep(0.2)
        sentiment = 'positive' if 'good' in content.lower() else 'neutral'

        return {
            'analysis_results': [{
                'type': 'sentiment',
                'result': sentiment,
                'confidence': 0.85
            }],
            'metadata': {
                'sentiment': sentiment,
                'sentiment_confidence': 0.85
            },
            'performance_stats': {
                'sentiment_duration': time.time() - start
            },
            'processing_log': ['Sentiment analysis completed']
        }

    def entity_extractor(state):
        """Extract entities with timing"""
        start = time.time()
        content = state.get('content', '')

        # Simulate entity extraction
        time.sleep(0.3)
        entities = ['Organization', 'Person'] if len(content) > 50 else ['Person']

        return {
            'analysis_results': [{
                'type': 'entities',
                'result': entities,
                'count': len(entities)
            }],
            'metadata': {
                'entity_count': len(entities),
                'entities': entities
            },
            'performance_stats': {
                'entity_duration': time.time() - start
            },
            'processing_log': ['Entity extraction completed']
        }

    def keyword_analyzer(state):
        """Extract keywords with timing"""
        start = time.time()
        content = state.get('content', '')

        # Simulate keyword extraction
        time.sleep(0.15)
        keywords = content.lower().split()[:5]  # Simple keyword extraction

        return {
            'analysis_results': [{
                'type': 'keywords',
                'result': keywords,
                'count': len(keywords)
            }],
            'metadata': {
                'keyword_count': len(keywords),
                'top_keywords': keywords[:3]
            },
            'performance_stats': {
                'keyword_duration': time.time() - start
            },
            'processing_log': ['Keyword analysis completed']
        }

    def quality_scorer(state):
        """Score content quality"""
        start = time.time()
        content = state.get('content', '')

        # Simulate quality scoring
        time.sleep(0.1)
        score = min(100, len(content) * 2)  # Simple scoring

        return {
            'analysis_results': [{
                'type': 'quality',
                'result': score,
                'grade': 'A' if score > 80 else 'B' if score > 60 else 'C'
            }],
            'metadata': {
                'quality_score': score,
                'quality_grade': 'A' if score > 80 else 'B' if score > 60 else 'C'
            },
            'performance_stats': {
                'quality_duration': time.time() - start
            },
            'processing_log': ['Quality scoring completed'],
            'final_score': score  # This will be the final score (set reducer)
        }

    def combine_analysis(state):
        """Combine all analysis results"""
        results = state.get('analysis_results', [])
        metadata = state.get('metadata', {})
        stats = state.get('performance_stats', {})
        total_time = time.time() - state.get('start_time', time.time())

        return {
            'final_analysis': {
                'total_analyses': len(results),
                'metadata_summary': metadata,
                'performance': {
                    **stats,
                    'total_duration': total_time,
                    'parallel_speedup': f"{sum(stats.values()):.2f}s sequential vs {total_time:.2f}s parallel"
                },
                'detailed_results': results,
                'overall_score': state.get('final_score', 0)
            },
            'processing_log': ['Analysis combination completed']
        }

    # Build the graph
    (graph
     .add_node('distributor', distribute_content)
     .add_node('sentiment_analyzer', sentiment_analyzer)
     .add_node('entity_extractor', entity_extractor)
     .add_node('keyword_analyzer', keyword_analyzer)
     .add_node('quality_scorer', quality_scorer)
     .add_node('combiner', combine_analysis)
     .set_entry_point('distributor')

     # All analyzers feed into combiner
     .add_edge('sentiment_analyzer', 'combiner')
     .add_edge('entity_extractor', 'combiner')
     .add_edge('keyword_analyzer', 'combiner')
     .add_edge('quality_scorer', 'combiner'))

    return graph.compile()

# Test the analyzer
analyzer = create_content_analyzer()
result = analyzer.invoke({
    'content': 'This is a good example of quality content with multiple organizations and people mentioned.'
})

print("📊 Analysis Results:")
print(f"Total analyses: {result['final_analysis']['total_analyses']}")
print(f"Metadata: {result['final_analysis']['metadata_summary']}")
print(f"Performance: {result['final_analysis']['performance']['parallel_speedup']}")
print(f"Overall score: {result['final_analysis']['overall_score']}")
print(f"Processing steps: {len(result['processing_log'])}")
```

**Output:**
```
📊 Analysis Results:
Total analyses: 4
Metadata: {'sentiment': 'positive', 'sentiment_confidence': 0.85, 'entity_count': 2, ...}
Performance: 0.75s sequential vs 0.30s parallel
Overall score: 180
Processing steps: 6
```

## 🔍 How Reducer Merging Works Internally

Understanding the merge process helps you design better state structures:

```python
# Internal merge process (simplified):
class State:
    def merge(self, updates, reducers):
        for field, new_value in updates.items():
            if field in reducers:
                reducer = reducers[field]
                current_value = self.data.get(field)

                if reducer == 'extend':
                    # Combine lists
                    current_list = current_value or []
                    new_list = new_value if isinstance(new_value, list) else [new_value]
                    self.data[field] = current_list + new_list

                elif reducer == 'append':
                    # Append single item
                    current_list = current_value or []
                    self.data[field] = current_list + [new_value]

                elif reducer == 'merge':
                    # Deep merge dictionaries
                    current_dict = current_value or {}
                    if isinstance(new_value, dict):
                        self.data[field] = {**current_dict, **new_value}

                elif reducer == 'set':
                    # Simple replacement
                    self.data[field] = new_value

                elif callable(reducer):
                    # Custom reducer function
                    self.data[field] = reducer(current_value, new_value)
            else:
                # No reducer - simple set
                self.data[field] = new_value
```

## 🎯 Best Practices

### 1. **Choose the Right Reducer**
```python
# ✅ Good: Choose reducers that match your data patterns
graph = StateGraph(state_reducers={
    'results': 'extend',           # For collecting parallel results
    'user_profile': 'merge',       # For building complex objects
    'current_status': 'set',       # For status updates
    'processing_times': 'extend'   # For performance metrics
})

# ❌ Bad: Using wrong reducer for data type
graph = StateGraph(state_reducers={
    'results': 'set',             # Will lose parallel results!
    'status': 'extend'            # Will create status lists!
})
```

### 2. **Design State Structure for Parallel Updates**
```python
# ✅ Good: Separate fields for different types of updates
{
    'individual_results': [],      # Each node adds its result
    'shared_metadata': {},         # Nodes update different metadata fields
    'final_status': None,          # Only final node sets this
    'error_log': []               # Collect errors from any node
}

# ❌ Bad: Mixed update patterns in single field
{
    'everything': {}  # Hard to manage with reducers
}
```

### 3. **Handle Edge Cases**
```python
def safe_custom_reducer(current, new):
    """Handle None values and type mismatches"""
    if current is None:
        return new
    if new is None:
        return current

    # Ensure type compatibility
    if type(current) != type(new):
        return new  # Fallback to replacement

    # Your merge logic here
    return merged_value
```

### 4. **Test Reducer Behavior**
```python
# Test your reducers with edge cases
test_cases = [
    ({'results': []}, {'results': ['new']}),           # Empty + data
    ({'results': ['old']}, {'results': []}),           # Data + empty  
    ({'results': None}, {'results': ['new']}),         # None + data
    ({'metadata': {}}, {'metadata': {'key': 'val'}}),  # Empty + data
]

for current, new in test_cases:
    # Test each reducer with edge cases
    pass
```

## 🚀 Advanced State Patterns

### Pattern 1: Hierarchical State Organization
```python
graph = StateGraph(state_reducers={
    'processing.results': 'extend',
    'processing.errors': 'extend', 
    'processing.timing': 'merge',
    'analysis.sentiment': 'set',
    'analysis.entities': 'extend',
    'metadata.user': 'merge',
    'metadata.system': 'merge'
})
```

### Pattern 2: Conditional State Updates
```python
def conditional_reducer(current, new):
    """Only update if new value meets criteria"""
    if isinstance(new, dict) and new.get('priority', 0) > current.get('priority', 0):
        return new
    return current

graph = StateGraph(state_reducers={
    'best_result': conditional_reducer
})
```

### Pattern 3: State Validation
```python
def validated_reducer(current, new):
    """Validate before merging"""
    if not validate_data(new):
        return current  # Keep current if new is invalid

    return merge_safely(current, new)
```

## 🎓 Key Takeaways

1. **Reducers solve the parallel state update problem** by defining merge strategies
2. **Choose the right reducer** for your data type and update pattern
3. **Design state structure** to work well with parallel updates
4. **Test edge cases** to ensure robust merging behavior
5. **Use custom reducers** for complex merging logic

**Master state management, and you'll build robust parallel workflows that never lose data!** 🎯
```python
    # All data accessible from state
    # Updates returned as dict
    return {
        "conversation_history": state["conversation_history"] + [state["user_input"]],
        "user_context": {**state["user_context"], "last_input": state["user_input"]},
        "response": generate_response(state["user_input"])
    }
```

## Working with Dictionary States

In GraphFlow, state is managed using simple Python dictionaries. Your nodes receive state as a dictionary and return dictionary updates.

### Basic State Design

```python
def process_user_input(state):
    """Process user input - state is a dict"""
    user_input = state.get("user_input", "")

    return {
        "processed_input": user_input.strip().lower(),
        "input_length": len(user_input),
        "processing_timestamp": time.time()
    }
```

### State Structure Guidelines

Design your state dictionaries for clarity and ease of use:

```python
# Example state structure for a chat agent
initial_state = {
    # User interaction
    "user_input": "",
    "conversation_history": [],

    # Processing data
    "intent": None,
    "entities": {},
    "response": "",

    # Control flow
    "step": "input",
    "retry_count": 0,
    "session_id": None
}
```

### Advanced State Patterns

**Nested Data for Complex Use Cases:**
```python
def initialize_complex_state():
    return {
        # Core conversation data
        "current_input": "",
        "conversation_history": [],

        # User information
        "user_profile": {
            "name": "",
            "preferences": [],
            "interaction_count": 0
        },

        # Processing state
        "current_intent": "",
        "processing_stage": "input",
        "confidence_score": 0.0,

        # Control flow
        "should_continue": True,
        "error_occurred": False,
        "debug_info": {}
    }
```

**State for Different Workflow Types:**

```python
# Data Processing Pipeline State
def create_processing_state(raw_data):
    return {
        "raw_data": raw_data,
        "cleaned_data": None,
        "processed_data": None,
        "results": None,
        "processing_stage": "start",
        "error_details": None
    }

# Multi-Agent Reasoning State  
def create_reasoning_state(problem):
    return {
        "problem_statement": problem,
        "gathered_facts": [],
        "hypotheses": [],
        "evidence": [],
        "reasoning_steps": [],
        "conclusion": None,
        "confidence_level": 0.0
    }
```

## State Update Patterns

### Simple Updates

```python
def simple_processor(state: MyState) -> dict:
    """Return a dictionary of updates."""
    return {
        "processed": True,
        "result": f"Processed: {state['input']}",
        "timestamp": datetime.now().isoformat()
    }
```

### Conditional Updates

```python
def conditional_processor(state: MyState) -> dict:
    """Update different fields based on conditions."""
    updates = {
        "step_count": state["step_count"] + 1
    }

    if state["step_count"] > 5:
        updates["status"] = "completed"
        updates["final_result"] = "Process finished"
    else:
        updates["status"] = "in_progress"
        updates["next_action"] = "continue_processing"

    return updates
```

### Accumulating Updates

```python
def accumulating_processor(state: MyState) -> dict:
    """Build up data over multiple iterations."""
    current_results = state.get("accumulated_results", [])
    new_result = process_current_input(state["current_input"])

    return {
        "accumulated_results": current_results + [new_result],
        "total_processed": len(current_results) + 1,
        "last_processed": new_result
    }
```

### Safe State Updates

```python
def safe_list_append(state: MyState) -> dict:
    """Safely append to lists without mutation."""
    # DON'T do this (mutates original state):
    # state["items"].append(new_item)  # BAD!

    # DO this (creates new list):
    current_items = state.get("items", [])
    new_items = current_items + [new_item]  # GOOD!

    return {"items": new_items}

def safe_dict_update(state: MyState) -> dict:
    """Safely update nested dictionaries."""
    # DON'T do this:
    # state["config"]["setting"] = new_value  # BAD!

    # DO this:
    current_config = state.get("config", {})
    updated_config = {**current_config, "setting": new_value}  # GOOD!

    return {"config": updated_config}
```

## State Flow Patterns

### Linear Flow

```python
def create_linear_state(input_data):
    return {
        "input_data": input_data,
        "step1_result": None,
        "step2_result": None, 
        "step3_result": None,
        "final_output": None
    }

def step1(state):
    result = f"Step1: {state['input_data']}"
    return {"step1_result": result}

def step2(state):
    result = f"Step2: {state['step1_result']}"
    return {"step2_result": result}

def step3(state):
    result = f"Step3: {state['step2_result']}"
    return {"step3_result": result, "final_output": result}
```

### Branching Flow

```python
def create_branching_state(input_data):
    return {
        "input_type": "",
        "input_data": input_data,
        "path_taken": "",
        "text_result": None,
        "number_result": None,
        "image_result": None
    }

def classifier(state):
    if isinstance(state["input_data"], str):
        return {"input_type": "text", "path_taken": "text_processing"}
    elif isinstance(state["input_data"], (int, float)):
        return {"input_type": "number", "path_taken": "number_processing"}
    else:
        return {"input_type": "other", "path_taken": "general_processing"}

def route_by_type(state):
    return f"{state['input_type']}_processor"
```

### Accumulating Flow

```python
def create_accumulating_state(items):
    return {
        "items_to_process": items,
        "current_index": 0,
        "processed_items": [],
        "accumulated_result": "",
        "is_complete": False
    }

def process_next_item(state):
    items = state["items_to_process"]
    index = state["current_index"]

    if index >= len(items):
        return Command(
            update={"is_complete": True},
            goto=END
        )

    current_item = items[index]
    processed = f"Processed: {current_item}"
    new_accumulated = state["accumulated_result"] + f" | {processed}"

    return Command(
        update={
            "current_index": index + 1,
            "processed_items": state["processed_items"] + [processed],
            "accumulated_result": new_accumulated
        },
        goto="process_next_item"  # Loop back
    )
```

## State Validation and Type Safety

### Runtime Validation

```python
def validate_state(state: MyState) -> None:
    """Validate state before processing."""
    required_fields = ["user_input", "step_count"]

    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(state["step_count"], int):
        raise TypeError("step_count must be an integer")

    if state["step_count"] < 0:
        raise ValueError("step_count cannot be negative")

def validated_processor(state: MyState) -> dict:
    validate_state(state)
    # ... process with confidence
    return {"result": "validated processing complete"}
```

### Runtime Validation

```python
def validate_state(state):
    """Validate state structure at runtime."""
    required_fields = ["step_count", "input_data"]

    for field in required_fields:
        if field not in state:
            raise KeyError(f"Missing required field: {field}")

    if not isinstance(state["step_count"], int):
        raise TypeError("step_count must be an integer")

    if state["step_count"] < 0:
        raise ValueError("step_count cannot be negative")

def validated_processor(state):
    validate_state(state)
    # ... process with confidence
    return {"result": "validated processing complete"}
```

### Default Value Patterns

```python
def processor_with_defaults(state):
    """Handle missing optional fields gracefully."""

    # Use get() with defaults for optional fields
    user_name = state.get("user_name", "Anonymous")
    preferences = state.get("preferences", [])
    config = state.get("config", {"theme": "default"})

    # Process with safe defaults
    result = f"Hello {user_name}! You have {len(preferences)} preferences."

    return {
        "greeting": result,
        "user_name": user_name,  # Ensure it's set for future nodes
        "preferences": preferences,
        "config": config
    }
```

## Common State Management Patterns

### State Machine Pattern

```python
def create_state_machine(initial_state="idle"):
    return {
        "current_state": initial_state,
        "valid_transitions": {
            "idle": ["processing"],
            "processing": ["completed", "error"],
            "error": ["idle", "processing"],
            "completed": ["idle"]
        },
        "state_data": {},
        "transition_history": []
    }

def state_machine_processor(state):
    current = state["current_state"]
    transitions = state["valid_transitions"]

    # Determine next state based on logic
    if current == "idle" and should_start_processing():
        next_state = "processing"
    elif current == "processing" and is_complete():
        next_state = "completed"
    elif current == "processing" and has_error():
        next_state = "error"
    else:
        next_state = current  # Stay in current state

    # Validate transition
    if next_state not in transitions.get(current, []):
        return Command(
            update={"error": f"Invalid transition: {current} -> {next_state}"},
            goto="error_handler"
        )

    return Command(
        update={
            "current_state": next_state,
            "transition_history": state["transition_history"] + [f"{current}->{next_state}"]
        },
        goto=next_state
    )
```

### Cache Pattern

```python
def create_cached_state():
    return {
        "cache": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "current_query": ""
    }

def cached_processor(state):
    query = state["current_query"]
    cache = state["cache"]

    # Check cache first
    if query in cache:
        return {
            "result": cache[query],
            "cache_hits": state["cache_hits"] + 1,
            "from_cache": True
        }

    # Compute result
    result = expensive_computation(query)

    # Update cache
    updated_cache = {**cache, query: result}

    return {
        "result": result,
        "cache": updated_cache,
        "cache_misses": state["cache_misses"] + 1,
        "from_cache": False
    }
```

### Error Handling Pattern

```python
def create_error_handling_state():
    return {
        "success": True,
        "error_message": None,
        "error_code": None,
        "retry_count": 0,
        "max_retries": 3,
        "last_successful_step": None
    }

def error_aware_processor(state):
    try:
        result = risky_operation(state["input_data"])

        return Command(
            update={
                "success": True,
                "result": result,
                "last_successful_step": "risky_operation",
                "retry_count": 0  # Reset on success
            },
            goto="next_step"
        )

    except Exception as e:
        retry_count = state["retry_count"] + 1

        if retry_count >= state["max_retries"]:
            return Command(
                update={
                    "success": False,
                    "error_message": str(e),
                    "error_code": "MAX_RETRIES_EXCEEDED",
                    "retry_count": retry_count
                },
                goto="error_handler"
            )
        else:
            return Command(
                update={
                    "success": False,
                    "error_message": str(e),
                    "retry_count": retry_count
                },
                goto="retry_operation"
            )
```

## Performance Considerations

### Memory-Efficient State

```python
# AVOID: Storing large objects in state
def bad_state_example():
    return {
        "huge_dataframe": df,  # Don't store large DataFrames in state
        "entire_file_content": file_content,  # Don't store large files
        "full_image_data": image_bytes  # Don't store binary data
    }

# PREFER: Store references and metadata
def good_state_example():
    return {
        "dataframe_path": "/path/to/data.csv",  # Store file path instead
        "file_metadata": {"size": 1024, "lines": 100},  # Store summary info
        "image_id": "img_123",  # Store identifier
        "processing_results": {"score": 0.95}  # Store computed results
    }
```

### Efficient Updates

```python
def efficient_list_update(state):
    """Efficient ways to update lists."""
    current_items = state.get("items", [])

    # For small lists: create new list
    if len(current_items) < 100:
        return {"items": current_items + [new_item]}

    # For large lists: consider alternatives
    return {
        "items": current_items[-99:] + [new_item],  # Keep only recent items
        "total_items_processed": state.get("total_items_processed", 0) + 1
    }

def efficient_dict_update(state: MyState) -> dict:
    """Efficient dictionary updates."""
    current_data = state.get("data", {})

    # Shallow merge for small dicts
    return {"data": {**current_data, "new_key": "new_value"}}

    # For deep merging, consider using a utility function
    # return {"data": deep_merge(current_data, new_data)}
```

## Testing State Management

### Unit Testing State Updates

```python
import unittest

class TestStateManagement(unittest.TestCase):

    def test_simple_update(self):
        """Test basic state updates."""
        initial_state = {"counter": 0, "items": []}

        def increment_counter(state):
            return {"counter": state["counter"] + 1}

        result = increment_counter(initial_state)
        expected = {"counter": 1}

        self.assertEqual(result, expected)

    def test_state_immutability(self):
        """Test that original state isn't modified."""
        initial_state = {"counter": 0, "items": ["a", "b"]}
        original_items = initial_state["items"]

        def add_item(state: TestState) -> dict:
            return {"items": state["items"] + ["c"]}

        result = add_item(initial_state)

        # Original state should be unchanged
        self.assertEqual(initial_state["items"], ["a", "b"])
        self.assertIs(initial_state["items"], original_items)

        # Result should have new list
        self.assertEqual(result["items"], ["a", "b", "c"])

    def test_conditional_updates(self):
        """Test conditional state updates."""
        def conditional_processor(state: TestState) -> dict:
            if state["counter"] >= 5:
                return {"status": "complete", "counter": state["counter"]}
            else:
                return {"status": "in_progress", "counter": state["counter"] + 1}

        # Test in-progress case
        result1 = conditional_processor({"counter": 3, "items": []})
        self.assertEqual(result1, {"status": "in_progress", "counter": 4})

        # Test completion case
        result2 = conditional_processor({"counter": 5, "items": []})
        self.assertEqual(result2, {"status": "complete", "counter": 5})

if __name__ == "__main__":
    unittest.main()
```

## Best Practices Summary

### Do's ✅

1. **Use simple dictionaries** for all state structures
2. **Return new objects** instead of mutating existing ones
3. **Validate required fields** in node functions  
4. **Use None for optional fields** that might not exist
5. **Keep state minimal** - store only what you need
6. **Use descriptive field names** like `user_input` not `data`
7. **Group related fields** in nested dictionary structures
8. **Handle missing fields gracefully** with `.get()` and defaults

### Don'ts ❌

1. **Don't mutate state directly** - always return updates
2. **Don't store large objects** in state - use references
3. **Don't use overly nested structures** - keep it flat when possible
4. **Don't ignore field validation** - check types when needed
5. **Don't make everything optional** - be explicit about requirements
6. **Don't mix data types** in the same field inconsistently
7. **Don't forget error handling** for state validation
8. **Don't store functions or objects** that can't be serialized

### Quick Reference

```python
# Perfect state management example

def create_well_designed_state(user_id, session_id, input_text):
    return {
        # Required fields (core data)
        "user_id": user_id,
        "session_id": session_id,
        "input_text": input_text,

        # Processing state
        "current_step": "start",
        "completed_steps": [],

        # Results (optional until computed)
        "processed_result": None,
        "confidence_score": None,

        # Metadata
        "metadata": {},
        "error_info": None
    }

def well_designed_processor(state):
    # Validate required fields
    if not state.get("input_text", "").strip():
        return {
            "error_info": {"code": "EMPTY_INPUT", "message": "Input text is required"}
        }

    # Process with confidence
    result = process_text(state["input_text"])

    # Return clean updates
    return {
        "processed_result": result,
        "confidence_score": 0.95,
        "current_step": "completed",
        "completed_steps": state["completed_steps"] + ["text_processing"],
        "metadata": {
            **state.get("metadata", {}),
            "processed_at": datetime.now().isoformat()
        }
    }
```