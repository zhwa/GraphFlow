# Chapter 5: State Management
**Mastering TypedDict schemas and state flow patterns**

State is the heart of GraphFlow - every node processes state, every decision is based on state, and every output is state. This chapter will teach you how to design effective state schemas and manage state flow like a pro.

## Understanding State in GraphFlow

### What is State?

In GraphFlow, state is a Python dictionary that flows through your graph. Each node receives the current state and can update it. Think of state as the "memory" of your workflow - it holds all the data your nodes need to make decisions and perform work.

```python
# Example state flowing through a workflow
initial_state = {
    "user_input": "Hello, how are you?",
    "conversation_history": [],
    "user_context": {},
    "current_step": "analyze_input"
}

# After processing through various nodes...
final_state = {
    "user_input": "Hello, how are you?",
    "conversation_history": [
        {"user": "Hello, how are you?", "bot": "Hi! I'm doing well, thanks for asking!"}
    ],
    "user_context": {"greeting_given": True, "mood": "friendly"},
    "current_step": "completed",
    "response": "Hi! I'm doing well, thanks for asking!"
}
```

### State vs Variables

**Traditional Programming:**
```python
# Variables scattered across functions
def process_user_input(input_text, history, context):
    # Process input
    updated_history = history + [input_text]
    updated_context = {**context, "last_input": input_text}
    return updated_history, updated_context, response

# Hard to track what data exists where
```

**GraphFlow State Approach:**
```python
# Everything centralized in state
def process_user_input(state: ConversationState) -> dict:
    # All data accessible from state
    # Updates returned as dict
    return {
        "conversation_history": state["conversation_history"] + [state["user_input"]],
        "user_context": {**state["user_context"], "last_input": state["user_input"]},
        "response": generate_response(state["user_input"])
    }
```

## Designing State Schemas with TypedDict

### Basic Schema Design

```python
from typing import TypedDict, List, Optional, Dict, Any

class BasicAgentState(TypedDict):
    # Required fields (no Optional)
    user_input: str
    messages: List[str]
    step_count: int
    
    # Optional fields
    response: Optional[str]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]
```

### Advanced Schema Patterns

**Nested State for Complex Data:**
```python
class UserProfile(TypedDict):
    name: str
    preferences: List[str]
    interaction_count: int

class ConversationTurn(TypedDict):
    timestamp: str
    user_message: str
    bot_response: str
    confidence: float

class AdvancedChatState(TypedDict):
    # Core conversation data
    current_input: str
    conversation_history: List[ConversationTurn]
    
    # User information
    user_profile: Optional[UserProfile]
    
    # Processing state
    current_intent: str
    processing_stage: str
    confidence_score: float
    
    # Control flow
    should_continue: bool
    error_occurred: bool
    debug_info: Dict[str, Any]
```

**State for Different Workflow Types:**

```python
# Sequential Processing State
class DataProcessingState(TypedDict):
    raw_data: Dict[str, Any]
    cleaned_data: Optional[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    processing_stage: str
    error_details: Optional[str]

# Agent Reasoning State  
class ReasoningAgentState(TypedDict):
    problem_statement: str
    gathered_facts: List[str]
    hypotheses: List[str]
    evidence: List[str]
    reasoning_steps: List[str]
    conclusion: Optional[str]
    confidence_level: float

# Multi-step Workflow State
class WorkflowState(TypedDict):
    input_data: Dict[str, Any]
    step_results: Dict[str, Any]  # Results keyed by step name
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    overall_status: str
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
class LinearState(TypedDict):
    input_data: str
    step1_result: Optional[str]
    step2_result: Optional[str] 
    step3_result: Optional[str]
    final_output: Optional[str]

def step1(state: LinearState) -> dict:
    result = f"Step1: {state['input_data']}"
    return {"step1_result": result}

def step2(state: LinearState) -> dict:
    result = f"Step2: {state['step1_result']}"
    return {"step2_result": result}

def step3(state: LinearState) -> dict:
    result = f"Step3: {state['step2_result']}"
    return {"step3_result": result, "final_output": result}
```

### Branching Flow

```python
class BranchingState(TypedDict):
    input_type: str
    input_data: Any
    path_taken: str
    text_result: Optional[str]
    number_result: Optional[float]
    image_result: Optional[Dict[str, Any]]

def classifier(state: BranchingState) -> dict:
    if isinstance(state["input_data"], str):
        return {"input_type": "text", "path_taken": "text_processing"}
    elif isinstance(state["input_data"], (int, float)):
        return {"input_type": "number", "path_taken": "number_processing"}
    else:
        return {"input_type": "other", "path_taken": "general_processing"}

def route_by_type(state: BranchingState) -> str:
    return f"{state['input_type']}_processor"
```

### Accumulating Flow

```python
class AccumulatingState(TypedDict):
    items_to_process: List[str]
    current_index: int
    processed_items: List[str]
    accumulated_result: str
    is_complete: bool

def process_next_item(state: AccumulatingState) -> Command:
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

### Type Checking with mypy

```python
# Enable strict type checking
# mypy --strict your_graphflow_app.py

from typing import TypedDict, Optional, List

class StrictState(TypedDict):
    required_string: str
    required_number: int
    optional_list: Optional[List[str]]

def type_safe_processor(state: StrictState) -> dict:
    # mypy will catch type errors here
    name: str = state["required_string"]  # ✓ Correct
    count: int = state["required_number"]  # ✓ Correct
    
    # These would cause mypy errors:
    # wrong_type: int = state["required_string"]  # ✗ Type error
    # missing_field = state["nonexistent"]        # ✗ Key error
    
    return {"processed": f"{name} ({count})"}
```

### Default Value Patterns

```python
def processor_with_defaults(state: MyState) -> dict:
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
class StateMachineState(TypedDict):
    current_state: str
    valid_transitions: Dict[str, List[str]]
    state_data: Dict[str, Any]
    transition_history: List[str]

def state_machine_processor(state: StateMachineState) -> Command:
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
class CachedState(TypedDict):
    cache: Dict[str, Any]
    cache_hits: int
    cache_misses: int
    current_query: str

def cached_processor(state: CachedState) -> dict:
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
class ErrorHandlingState(TypedDict):
    success: bool
    error_message: Optional[str]
    error_code: Optional[str]
    retry_count: int
    max_retries: int
    last_successful_step: Optional[str]

def error_aware_processor(state: ErrorHandlingState) -> Command:
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
class BadState(TypedDict):
    huge_dataframe: Any  # Don't store large DataFrames in state
    entire_file_content: str  # Don't store large files
    full_image_data: bytes  # Don't store binary data

# PREFER: Store references and metadata
class GoodState(TypedDict):
    dataframe_path: str  # Store file path instead
    file_metadata: Dict[str, Any]  # Store summary info
    image_id: str  # Store identifier
    processing_results: Dict[str, float]  # Store computed results
```

### Efficient Updates

```python
def efficient_list_update(state: MyState) -> dict:
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
from typing import TypedDict

class TestState(TypedDict):
    counter: int
    items: List[str]

class TestStateManagement(unittest.TestCase):
    
    def test_simple_update(self):
        """Test basic state updates."""
        initial_state = {"counter": 0, "items": []}
        
        def increment_counter(state: TestState) -> dict:
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

1. **Use TypedDict** for all state schemas
2. **Return new objects** instead of mutating existing ones
3. **Validate required fields** in node functions
4. **Use Optional[]** for fields that might not exist
5. **Keep state minimal** - store only what you need
6. **Use descriptive field names** like `user_input` not `data`
7. **Group related fields** in nested TypedDict structures
8. **Handle missing fields gracefully** with `.get()` and defaults

### Don'ts ❌

1. **Don't mutate state directly** - always return updates
2. **Don't store large objects** in state - use references
3. **Don't use overly nested structures** - keep it flat when possible
4. **Don't ignore type hints** - they prevent bugs
5. **Don't make everything optional** - be explicit about requirements
6. **Don't mix data types** in the same field inconsistently
7. **Don't forget error handling** for state validation
8. **Don't store functions or objects** that can't be serialized

### Quick Reference

```python
# Perfect state management example
from typing import TypedDict, List, Optional, Dict, Any

class WellDesignedState(TypedDict):
    # Required fields (core data)
    user_id: str
    session_id: str
    input_text: str
    
    # Processing state
    current_step: str
    completed_steps: List[str]
    
    # Results (optional until computed)
    processed_result: Optional[str]
    confidence_score: Optional[float]
    
    # Metadata
    metadata: Dict[str, Any]
    error_info: Optional[Dict[str, str]]

def well_designed_processor(state: WellDesignedState) -> dict:
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

---

**Next: [Chapter 6: Nodes and Edges →](06-nodes-edges.md)**
