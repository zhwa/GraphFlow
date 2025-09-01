"""
GraphFlow Quick Start Example

This file demonstrates the basic usage of GraphFlow.
Run this file to see GraphFlow in action.
"""

import sys
import os
from typing import TypedDict, List

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END

# Define your state schema
class ChatState(TypedDict):
    messages: List[str]
    user_input: str
    turn_count: int

def greet_user(state: ChatState) -> dict:
    """Initial greeting node."""
    return {
        "messages": state["messages"] + ["Hello! How can I help you today?"],
        "turn_count": state["turn_count"] + 1
    }

def process_input(state: ChatState) -> Command:
    """Process user input and decide next action."""
    user_msg = state["user_input"].lower()
    
    # Generate response based on input
    if "bye" in user_msg or "goodbye" in user_msg:
        response = "Goodbye! Have a great day!"
        return Command(
            update={
                "messages": state["messages"] + [f"User: {state['user_input']}", response],
                "turn_count": state["turn_count"] + 1
            },
            goto=END
        )
    else:
        response = f"You said: '{state['user_input']}'. That's interesting!"
        return Command(
            update={
                "messages": state["messages"] + [f"User: {state['user_input']}", response],
                "turn_count": state["turn_count"] + 1
            },
            goto="ask_more"
        )

def ask_more(state: ChatState) -> dict:
    """Ask for more input."""
    return {
        "messages": state["messages"] + ["What else would you like to talk about?"]
    }

def main():
    """Main function to demonstrate GraphFlow."""
    print("GraphFlow Quick Start Example")
    print("=" * 40)
    
    # Build the graph
    graph = StateGraph(ChatState)
    graph.add_node("greet", greet_user)
    graph.add_node("process", process_input)
    graph.add_node("ask_more", ask_more)
    
    # Set up the flow
    graph.add_edge("greet", "process")
    graph.add_edge("ask_more", "process")
    graph.set_entry_point("greet")
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    # Test with different inputs
    test_inputs = [
        "Hello there!",
        "What's the weather like?",
        "Tell me a joke",
        "Goodbye!"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser Input: {user_input}")
        print("-" * 30)
        
        result = compiled_graph.invoke({
            "messages": [],
            "user_input": user_input,
            "turn_count": 0
        })
        
        print("Conversation:")
        for i, message in enumerate(result["messages"], 1):
            print(f"  {i}. {message}")
        
        print(f"Turn count: {result['turn_count']}")
        
        if "Goodbye" in result["messages"][-1]:
            print("\n(Conversation ended)")
            break

if __name__ == "__main__":
    main()
