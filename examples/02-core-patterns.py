"""
GraphFlow Examples

This module contains examples demonstrating various GraphFlow patterns
similar to LangGraph but built on PocketFlow.
"""

import sys
import os
from typing import TypedDict, List, Dict, Any
import json

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Send, Command, START, END

# Example 1: Simple Chat Agent
def example_simple_chat():
    """A simple chat agent that processes messages."""
    
    class ChatState(TypedDict):
        messages: List[str]
        user_input: str
        response: str
    
    def process_input(state: ChatState) -> Dict[str, Any]:
        """Process user input and generate a response."""
        user_msg = state.get("user_input", "")
        response = f"You said: {user_msg}. How can I help you?"
        
        return {
            "messages": state["messages"] + [user_msg, response],
            "response": response
        }
    
    # Build the graph
    graph = StateGraph(ChatState)
    graph.add_node("process", process_input)
    graph.set_entry_point("process")
    
    compiled = graph.compile()
    
    # Test the graph
    result = compiled.invoke({
        "messages": [],
        "user_input": "Hello there!",
        "response": ""
    })
    
    print("Simple Chat Example:")
    print(f"Messages: {result['messages']}")
    print(f"Response: {result['response']}\n")

# Example 2: Conditional Routing
def example_conditional_routing():
    """Demonstrate conditional routing based on state."""
    
    class RouterState(TypedDict):
        message: str
        route_taken: str
        processed: bool
    
    def analyze_message(state: RouterState) -> Dict[str, Any]:
        """Analyze the message and mark for routing."""
        message = state["message"].lower()
        
        if "urgent" in message:
            route = "urgent_handler"
        elif "question" in message or "?" in message:
            route = "qa_handler"
        else:
            route = "general_handler"
        
        return {"route_taken": route}
    
    def route_message(state: RouterState) -> str:
        """Route based on analysis."""
        return state["route_taken"]
    
    def urgent_handler(state: RouterState) -> Dict[str, Any]:
        """Handle urgent messages."""
        return {
            "message": f"URGENT: {state['message']}",
            "processed": True
        }
    
    def qa_handler(state: RouterState) -> Dict[str, Any]:
        """Handle questions."""
        return {
            "message": f"Q&A: {state['message']}",
            "processed": True
        }
    
    def general_handler(state: RouterState) -> Dict[str, Any]:
        """Handle general messages."""
        return {
            "message": f"General: {state['message']}",
            "processed": True
        }
    
    # Build the graph
    graph = StateGraph(RouterState)
    graph.add_node("analyze", analyze_message)
    graph.add_node("urgent_handler", urgent_handler)
    graph.add_node("qa_handler", qa_handler)
    graph.add_node("general_handler", general_handler)
    
    # Add conditional routing
    graph.add_conditional_edges("analyze", route_message)
    graph.set_entry_point("analyze")
    
    compiled = graph.compile()
    
    # Test different message types
    test_messages = [
        "This is urgent! Please help!",
        "What is the weather like today?",
        "Hello, how are you doing?"
    ]
    
    print("Conditional Routing Example:")
    for msg in test_messages:
        result = compiled.invoke({
            "message": msg,
            "route_taken": "",
            "processed": False
        })
        print(f"Input: {msg}")
        print(f"Route: {result['route_taken']}")
        print(f"Output: {result['message']}")
        print(f"Processed: {result['processed']}\n")

# Example 3: Using Command for Complex Routing
def example_command_routing():
    """Demonstrate Command objects for combining updates and routing."""
    
    class CommandState(TypedDict):
        count: int
        messages: List[str]
        done: bool
    
    def counter_node(state: CommandState) -> Command:
        """Increment counter and decide next action."""
        new_count = state["count"] + 1
        
        if new_count < 3:
            return Command(
                update={
                    "count": new_count,
                    "messages": state["messages"] + [f"Count: {new_count}"]
                },
                goto="counter"  # Loop back
            )
        else:
            return Command(
                update={
                    "count": new_count,
                    "messages": state["messages"] + [f"Final count: {new_count}"],
                    "done": True
                },
                goto=END
            )
    
    # Build the graph
    graph = StateGraph(CommandState)
    graph.add_node("counter", counter_node)
    graph.set_entry_point("counter")
    
    compiled = graph.compile()
    
    # Test the command routing
    result = compiled.invoke({
        "count": 0,
        "messages": [],
        "done": False
    })
    
    print("Command Routing Example:")
    print(f"Final count: {result['count']}")
    print(f"Messages: {result['messages']}")
    print(f"Done: {result['done']}\n")

# Example 4: Map-Reduce with Send
def example_map_reduce():
    """Demonstrate map-reduce pattern using Send objects."""
    
    class MapReduceState(TypedDict):
        items: List[str]
        processed_items: List[str]
        final_result: str
    
    def map_items(state: MapReduceState) -> List[Send]:
        """Map items to individual processing nodes."""
        sends = []
        for item in state["items"]:
            sends.append(Send("process_item", {"item": item}))
        return sends
    
    def process_item(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        item = state["item"]
        processed = f"Processed: {item.upper()}"
        return {"processed_item": processed}
    
    def reduce_results(state: MapReduceState) -> Dict[str, Any]:
        """Combine all processed items."""
        final = " | ".join(state["processed_items"])
        return {"final_result": final}
    
    # This is a simplified version - full map-reduce would require
    # more sophisticated parallel execution handling
    print("Map-Reduce Example (Simplified):")
    
    # Simulate the map-reduce process
    items = ["apple", "banana", "cherry"]
    processed_items = []
    
    for item in items:
        result = process_item({"item": item})
        processed_items.append(result["processed_item"])
    
    final_result = reduce_results({
        "items": items,
        "processed_items": processed_items,
        "final_result": ""
    })
    
    print(f"Items: {items}")
    print(f"Processed: {processed_items}")
    print(f"Final: {final_result['final_result']}\n")

# Example 5: Agent-like Workflow
def example_agent_workflow():
    """Demonstrate an agent-like workflow with tool calling."""
    
    class AgentState(TypedDict):
        user_query: str
        thoughts: List[str]
        actions: List[str]
        final_answer: str
        max_iterations: int
        current_iteration: int
    
    def think(state: AgentState) -> Dict[str, Any]:
        """Agent thinking step."""
        query = state["user_query"]
        iteration = state["current_iteration"]
        
        if "weather" in query.lower():
            thought = f"Iteration {iteration}: User asking about weather, need to call weather tool"
            action = "call_weather_tool"
        elif "time" in query.lower():
            thought = f"Iteration {iteration}: User asking about time, need to get current time"
            action = "get_time"
        else:
            thought = f"Iteration {iteration}: General query, can answer directly"
            action = "answer_directly"
        
        return {
            "thoughts": state["thoughts"] + [thought],
            "actions": state["actions"] + [action],
            "current_iteration": iteration + 1
        }
    
    def should_continue(state: AgentState) -> str:
        """Decide whether to continue or finish."""
        if state["current_iteration"] >= state["max_iterations"]:
            return "finish"
        
        last_action = state["actions"][-1] if state["actions"] else ""
        
        if last_action == "answer_directly":
            return "finish"
        else:
            return "execute_action"
    
    def execute_action(state: AgentState) -> Dict[str, Any]:
        """Execute the planned action."""
        last_action = state["actions"][-1] if state["actions"] else ""
        
        if last_action == "call_weather_tool":
            result = "Weather tool result: Sunny, 75°F"
        elif last_action == "get_time":
            result = "Current time: 2:30 PM"
        else:
            result = "Action executed"
        
        return {
            "thoughts": state["thoughts"] + [f"Action result: {result}"]
        }
    
    def finish(state: AgentState) -> Dict[str, Any]:
        """Generate final answer."""
        query = state["user_query"]
        thoughts = state["thoughts"]
        
        # Simple answer generation based on thoughts
        if any("weather" in t.lower() for t in thoughts):
            answer = "The weather is sunny and 75°F."
        elif any("time" in t.lower() for t in thoughts):
            answer = "The current time is 2:30 PM."
        else:
            answer = f"I've thought about your query: {query}"
        
        return {"final_answer": answer}
    
    # Build the agent graph
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("execute_action", execute_action)
    graph.add_node("finish", finish)
    
    # Add routing
    graph.add_conditional_edges("think", should_continue)
    graph.add_edge("execute_action", "think")  # Loop back to think
    graph.set_entry_point("think")
    
    compiled = graph.compile()
    
    # Test the agent
    test_queries = [
        "What's the weather like?",
        "What time is it?",
        "Tell me a joke"
    ]
    
    print("Agent Workflow Example:")
    for query in test_queries:
        result = compiled.invoke({
            "user_query": query,
            "thoughts": [],
            "actions": [],
            "final_answer": "",
            "max_iterations": 3,
            "current_iteration": 0
        })
        
        print(f"Query: {query}")
        print(f"Thoughts: {result['thoughts']}")
        print(f"Actions: {result['actions']}")
        print(f"Answer: {result['final_answer']}\n")

def run_all_examples():
    """Run all the examples."""
    print("=" * 60)
    print("GraphFlow Examples - LangGraph-like Agent Framework")
    print("=" * 60)
    print()
    
    example_simple_chat()
    example_conditional_routing()
    example_command_routing()
    example_map_reduce()
    example_agent_workflow()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()
