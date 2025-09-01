"""
GraphFlow Examples and Tests

This comprehensive file combines examples with testing to demonstrate
GraphFlow functionality while validating that everything works correctly.

Run this file to:
1. Test the framework basics
2. See working examples
3. Validate your installation
"""

import sys
import os
from typing import TypedDict, List, Dict, Any

import sys
import os
from typing import TypedDict, List, Dict, Any

import sys
import os
from typing import TypedDict, List, Dict, Any

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from graphflow import StateGraph, Command, END
        print("✓ GraphFlow core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Make sure graphflow.py is in the parent directory")
        return False

def example_basic_counter():
    """Example 1: Basic counter with state updates."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Counter")
    print("="*50)
    
    try:
        from graphflow import StateGraph
        
        class CounterState(TypedDict):
            count: int
            messages: List[str]
        
        def increment(state: CounterState) -> dict:
            new_count = state["count"] + 1
            return {
                "count": new_count,
                "messages": state["messages"] + [f"Count is now: {new_count}"]
            }
        
        # Build graph
        graph = StateGraph(CounterState)
        graph.add_node("increment", increment)
        graph.set_entry_point("increment")
        app = graph.compile()
        
        # Test it
        result = app.invoke({"count": 0, "messages": []})
        
        print(f"Initial count: 0")
        print(f"Final count: {result['count']}")
        print(f"Messages: {result['messages']}")
        
        # Validate
        assert result["count"] == 1, f"Expected count=1, got {result['count']}"
        assert len(result["messages"]) == 1, f"Expected 1 message, got {len(result['messages'])}"
        
        print("✓ Basic counter example passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic counter example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_conditional_routing():
    """Example 2: Conditional routing based on state."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Conditional Routing")
    print("="*50)
    
    try:
        from graphflow import StateGraph, END
        
        class RoutingState(TypedDict):
            number: int
            result: str
            path: str
        
        def analyze_number(state: RoutingState) -> dict:
            return {"path": "analyzed"}
        
        def route_by_number(state: RoutingState) -> str:
            if state["number"] > 10:
                return "high"
            elif state["number"] > 0:
                return "medium"
            else:
                return "low"
        
        def handle_high(state: RoutingState) -> dict:
            return {
                "result": f"{state['number']} is a high number!",
                "path": state["path"] + " -> high"
            }
        
        def handle_medium(state: RoutingState) -> dict:
            return {
                "result": f"{state['number']} is a medium number.",
                "path": state["path"] + " -> medium"
            }
        
        def handle_low(state: RoutingState) -> dict:
            return {
                "result": f"{state['number']} is a low number.",
                "path": state["path"] + " -> low"
            }
        
        # Build graph
        graph = StateGraph(RoutingState)
        graph.add_node("analyze", analyze_number)
        graph.add_node("high", handle_high)
        graph.add_node("medium", handle_medium)
        graph.add_node("low", handle_low)
        
        graph.add_conditional_edges("analyze", route_by_number, {
            "high": "high",
            "medium": "medium", 
            "low": "low"
        })
        graph.set_entry_point("analyze")
        app = graph.compile()
        
        # Test different numbers
        test_cases = [15, 5, -3]
        
        for number in test_cases:
            result = app.invoke({
                "number": number,
                "result": "",
                "path": ""
            })
            
            print(f"Number: {number}")
            print(f"  Result: {result['result']}")
            print(f"  Path: {result['path']}")
            
            # Basic validation
            assert result["result"] != "", "Result should not be empty"
            assert "analyzed" in result["path"], "Should have gone through analyze node"
        
        print("✓ Conditional routing example passed!")
        return True
        
    except Exception as e:
        print(f"✗ Conditional routing example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_chat_with_memory():
    """Example 3: Chat agent with conversation memory."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Chat Agent with Memory")
    print("="*50)
    
    try:
        from graphflow import StateGraph, Command, END
        
        class ChatState(TypedDict):
            messages: List[str]
            user_input: str
            turn_count: int
            conversation_active: bool
        
        def process_message(state: ChatState) -> Command:
            user_msg = state["user_input"].lower()
            turn = state["turn_count"] + 1
            
            # Check for exit conditions
            if "bye" in user_msg or "goodbye" in user_msg or turn > 3:
                response = "Goodbye! Thanks for chatting!"
                return Command(
                    update={
                        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"],
                        "turn_count": turn,
                        "conversation_active": False
                    },
                    goto=END
                )
            else:
                response = f"You said '{state['user_input']}'. Tell me more! (Turn {turn})"
                return Command(
                    update={
                        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"],
                        "turn_count": turn,
                        "conversation_active": True
                    },
                    goto="process"
                )
        
        # Build graph
        graph = StateGraph(ChatState)
        graph.add_node("process", process_message)
        graph.set_entry_point("process")
        app = graph.compile()
        
        # Test conversation
        test_messages = ["Hello!", "How are you?", "Tell me a story", "Goodbye!"]
        
        for i, message in enumerate(test_messages):
            print(f"\n--- Turn {i+1} ---")
            result = app.invoke({
                "messages": [],
                "user_input": message,
                "turn_count": 0,
                "conversation_active": True
            })
            
            print(f"Input: {message}")
            print("Conversation:")
            for msg in result["messages"]:
                print(f"  {msg}")
            
            if not result.get("conversation_active", True):
                print("(Conversation ended)")
                break
        
        print("✓ Chat with memory example passed!")
        return True
        
    except Exception as e:
        print(f"✗ Chat with memory example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_multi_step_workflow():
    """Example 4: Multi-step data processing workflow."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Multi-Step Workflow")
    print("="*50)
    
    try:
        from graphflow import StateGraph
        
        class WorkflowState(TypedDict):
            raw_data: str
            cleaned_data: str
            processed_data: str
            final_result: str
            steps_completed: List[str]
        
        def clean_data(state: WorkflowState) -> dict:
            cleaned = state["raw_data"].strip().lower()
            return {
                "cleaned_data": cleaned,
                "steps_completed": state["steps_completed"] + ["cleaned"]
            }
        
        def process_data(state: WorkflowState) -> dict:
            processed = f"processed_{state['cleaned_data']}"
            return {
                "processed_data": processed,
                "steps_completed": state["steps_completed"] + ["processed"]
            }
        
        def finalize_data(state: WorkflowState) -> dict:
            final = f"FINAL: {state['processed_data'].upper()}"
            return {
                "final_result": final,
                "steps_completed": state["steps_completed"] + ["finalized"]
            }
        
        # Build pipeline
        graph = StateGraph(WorkflowState)
        graph.add_node("clean", clean_data)
        graph.add_node("process", process_data)
        graph.add_node("finalize", finalize_data)
        
        graph.add_edge("clean", "process")
        graph.add_edge("process", "finalize")
        graph.set_entry_point("clean")
        app = graph.compile()
        
        # Test pipeline
        result = app.invoke({
            "raw_data": "  Hello World  ",
            "cleaned_data": "",
            "processed_data": "",
            "final_result": "",
            "steps_completed": []
        })
        
        print(f"Original: '  Hello World  '")
        print(f"Cleaned: '{result['cleaned_data']}'")
        print(f"Processed: '{result['processed_data']}'")
        print(f"Final: '{result['final_result']}'")
        print(f"Steps: {' -> '.join(result['steps_completed'])}")
        
        # Validate pipeline
        assert result["cleaned_data"] == "hello world", "Data should be cleaned"
        assert "processed_" in result["processed_data"], "Data should be processed"
        assert "FINAL:" in result["final_result"], "Data should be finalized"
        assert len(result["steps_completed"]) >= 3, "Should have at least 3 steps"
        
        print("✓ Multi-step workflow example passed!")
        return True
        
    except Exception as e:
        print(f"✗ Multi-step workflow example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_error_handling():
    """Example 5: Error handling and recovery."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Error Handling")
    print("="*50)
    
    try:
        from graphflow import StateGraph, END
        
        class ErrorState(TypedDict):
            input_value: int
            result: str
            error_count: int
            status: str
        
        def risky_operation(state: ErrorState) -> dict:
            value = state["input_value"]
            
            # Simulate an operation that might fail
            if value < 0:
                return {
                    "error_count": state["error_count"] + 1,
                    "status": "error",
                    "result": f"Error: Negative value {value} not allowed"
                }
            elif value == 0:
                return {
                    "error_count": state["error_count"] + 1,
                    "status": "warning",
                    "result": f"Warning: Zero value might cause issues"
                }
            else:
                return {
                    "status": "success",
                    "result": f"Successfully processed value: {value * 2}"
                }
        
        def route_by_status(state: ErrorState) -> str:
            return state["status"]
        
        def handle_error(state: ErrorState) -> dict:
            return {
                "result": state["result"] + " [ERROR HANDLED]",
                "status": "recovered"
            }
        
        def handle_warning(state: ErrorState) -> dict:
            return {
                "result": state["result"] + " [WARNING NOTED]",
                "status": "completed_with_warning"
            }
        
        def handle_success(state: ErrorState) -> dict:
            return {
                "result": state["result"] + " [SUCCESS]",
                "status": "completed"
            }
        
        # Build error-handling graph
        graph = StateGraph(ErrorState)
        graph.add_node("risky", risky_operation)
        graph.add_node("handle_error", handle_error)
        graph.add_node("handle_warning", handle_warning)
        graph.add_node("handle_success", handle_success)
        
        graph.add_conditional_edges("risky", route_by_status, {
            "error": "handle_error",
            "warning": "handle_warning",
            "success": "handle_success"
        })
        graph.set_entry_point("risky")
        app = graph.compile()
        
        # Test different scenarios
        test_cases = [-5, 0, 10]
        
        for value in test_cases:
            result = app.invoke({
                "input_value": value,
                "result": "",
                "error_count": 0,
                "status": ""
            })
            
            print(f"Input: {value}")
            print(f"  Status: {result['status']}")
            print(f"  Result: {result['result']}")
            print(f"  Errors: {result['error_count']}")
            
            # Validate error handling
            assert result["result"] != "", "Should have a result"
            assert result["status"] in ["recovered", "completed_with_warning", "completed"], "Should have valid status"
        
        print("✓ Error handling example passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error handling example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_examples():
    """Run all examples and tests."""
    print("GraphFlow Examples and Tests")
    print("=" * 60)
    print("This script demonstrates GraphFlow capabilities while testing functionality.")
    print("=" * 60)
    
    # Start with import test
    if not test_imports():
        print("\n❌ Cannot proceed - import test failed")
        return False
    
    # Run all examples
    examples = [
        ("Basic Counter", example_basic_counter),
        ("Conditional Routing", example_conditional_routing),
        ("Chat with Memory", example_chat_with_memory),
        ("Multi-Step Workflow", example_multi_step_workflow),
        ("Error Handling", example_error_handling)
    ]
    
    passed = 0
    total = len(examples)
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        if example_func():
            passed += 1
            print(f"✓ {name} completed successfully!")
        else:
            print(f"✗ {name} failed!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Examples passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 SUCCESS: All examples working correctly!")
        print("\nGraphFlow is ready to use. Try running individual examples:")
        print("  python examples/00-quick-start.py")
        print("  python examples/03-hello-world-qa.py")
        print("  python examples/02-core-patterns.py")
        return True
    else:
        print("❌ FAILURE: Some examples failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_examples()
    
    if not success:
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print("- Explore examples in the examples/ directory")
    print("- Read documentation in docs/")
    print("- Build your own GraphFlow applications!")
    print(f"{'='*60}")
