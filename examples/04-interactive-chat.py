"""
GraphFlow Example: Interactive Chat Agent with Real LLM
Ported from: PocketFlow cookbook/pocketflow-chat

This example demonstrates a conversational agent that:
- Maintains conversation history
- Handles user input/output  
- Uses REAL LLM calls (OpenAI, Anthropic, or Ollama)
- Loops until user says goodbye
- Uses Command objects for flow control

Setup:
1. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama
2. Run: python 04-interactive-chat.py

Original PocketFlow pattern:
- ChatNode with self-loop (node - "continue" >> node)
- User input in prep, LLM call in exec, response handling in post
- Loop control via return values

GraphFlow adaptation:
- State-based conversation history
- Real LLM integration with auto-provider detection
- Command objects for flow control  
- Cleaner separation of concerns
"""

import sys
import os
from typing import TypedDict, List, Dict

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, call_llm, configure_llm, get_llm_config

# State schema for chat conversation
class ChatState(TypedDict):
    messages: List[Dict[str, str]]  # Conversation history
    user_input: str                 # Current user input
    assistant_response: str         # Current assistant response
    should_continue: bool          # Whether to continue chatting
    turn_count: int               # Number of conversation turns

def setup_llm_node(state: ChatState) -> Command:
    """
    Configure LLM provider based on available credentials.
    """
    print("🔧 Setting up LLM provider...")
    
    # Try to auto-configure based on environment
    if os.environ.get("OPENAI_API_KEY"):
        configure_llm("openai", model="gpt-4")
        print("✅ Using OpenAI GPT-4")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        configure_llm("anthropic", model="claude-3-sonnet-20240229")
        print("✅ Using Anthropic Claude")
    else:
        # Try Ollama (local)
        try:
            configure_llm("ollama", model="llama2")
            # Test if Ollama is working
            call_llm("test")
            print("✅ Using Ollama (local)")
        except Exception:
            print("❌ No LLM provider available!")
            print("\nTo use this example, you need:")
            print("1. Set OPENAI_API_KEY environment variable, OR")
            print("2. Set ANTHROPIC_API_KEY environment variable, OR") 
            print("3. Start Ollama server: ollama serve")
            return Command(
                update={"should_continue": False},
                goto=END
            )
    
    config = get_llm_config()
    print(f"   Provider: {config['provider']}")
    print(f"   Model: {config['model']}")
    print("-" * 50)
    
    return Command(
        update={"should_continue": True},
        goto="get_input"
    )

def get_user_input(state: ChatState) -> Command:
    """
    Get input from the user and decide whether to continue.
    
    Equivalent to PocketFlow ChatNode.prep() method.
    """
    # Initialize on first run
    if state["turn_count"] == 0:
        print("Welcome to GraphFlow Chat with Real LLM!")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        print("-" * 50)
    
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Check for exit conditions
    if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
        return Command(
            update={
                "user_input": user_input,
                "should_continue": False
            },
            goto="farewell"
        )
    
    if not user_input:
        print("Please enter a message.")
        return Command(
            update={"should_continue": True},
            goto="get_input"  # Loop back for valid input
        )
    
    # Valid input - proceed to generate response
    return Command(
        update={
            "user_input": user_input,
            "should_continue": True,
            "turn_count": state["turn_count"] + 1
        },
        goto="generate_response"
    )

def generate_response(state: ChatState) -> Command:
    """
    Generate assistant response using real LLM.
    
    Equivalent to PocketFlow ChatNode.exec() method.
    """
    print("🧠 Thinking...")
    
    user_input = state["user_input"]
    messages = state["messages"]
    
    # Build conversation context for LLM
    conversation_messages = []
    
    # Add system message for context
    conversation_messages.append({
        "role": "system", 
        "content": "You are a helpful AI assistant. Be conversational and friendly."
    })
    
    # Add conversation history
    conversation_messages.extend(messages)
    
    # Add current user input
    conversation_messages.append({
        "role": "user",
        "content": user_input
    })
    
    try:
        # Call the real LLM
        response = call_llm(conversation_messages)
        
        return Command(
            update={"assistant_response": response},
            goto="update_conversation"
        )
        
    except Exception as e:
        error_response = f"Sorry, I encountered an error: {str(e)}"
        print(f"⚠️  LLM Error: {e}")
        
        return Command(
            update={"assistant_response": error_response},
            goto="update_conversation"
        )

def update_conversation(state: ChatState) -> Command:
    """
    Update conversation history and decide next action.
    
    Equivalent to PocketFlow ChatNode.post() method.
    """
    # Add messages to history
    new_messages = state["messages"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": state["assistant_response"]}
    ]
    
    # Display the response
    print(f"Assistant: {state['assistant_response']}")
    
    # Continue the conversation
    return Command(
        update={
            "messages": new_messages,
            "should_continue": True
        },
        goto="get_input"  # Loop back for next input
    )

def farewell(state: ChatState) -> dict:
    """Handle conversation ending."""
    print(f"\nAssistant: Goodbye! Thanks for chatting. We had {state['turn_count']} turns of conversation.")
    return Command(
        update={
            "assistant_response": "Goodbye! Thanks for chatting.",
            "should_continue": False
        },
        goto=END
    )

def build_chat_graph():
    """Build the interactive chat graph with LLM integration."""
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("setup_llm", setup_llm_node)
    graph.add_node("get_input", get_user_input)
    graph.add_node("generate_response", generate_response)
    graph.add_node("update_conversation", update_conversation)
    graph.add_node("farewell", farewell)
    
    # Set up flow
    graph.add_edge("generate_response", "update_conversation")
    graph.set_entry_point("setup_llm")
    
    return graph.compile()

def main():
    """Main function - start the chat session with real LLM."""
    print("🤖 GraphFlow Interactive Chat with Real LLM")
    print("=" * 50)
    
    # Build the chat graph
    chat_app = build_chat_graph()
    
    # Initialize conversation state
    initial_state = {
        "messages": [],
        "user_input": "",
        "assistant_response": "",
        "should_continue": True,
        "turn_count": 0
    }
    
    # Start the conversation
    try:
        chat_app.invoke(initial_state)
    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Chat session ended.")

if __name__ == "__main__":
    main()

"""
Key GraphFlow Improvements over PocketFlow:

1. Real LLM Integration:
   - Supports OpenAI, Anthropic, and Ollama
   - Auto-detects available providers
   - Handles API errors gracefully

2. Clearer State Management:
   - Explicit conversation history structure
   - Type-safe state updates with Commands
   - Turn counting and session management

3. Better Flow Control:
   - Command objects make routing explicit
   - Clear separation of setup/input/processing/output
   - Easy to add new conversation branches

4. Enhanced Error Handling:
   - Graceful handling of empty inputs
   - LLM API error recovery
   - Keyboard interrupt support

5. Production Ready:
   - Real conversations with context
   - Proper conversation memory
   - Professional user experience

Usage:
1. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama
2. Run: python 04-interactive-chat.py
3. Chat with a real AI assistant!

To extend this example:
1. Add conversation memory across sessions
2. Implement different AI personalities
3. Add multi-modal support (images, files)
4. Integrate with external APIs for richer responses
"""
