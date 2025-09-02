#!/usr/bin/env python3
"""
GraphFlow LLM Integration Example

This example shows how to use GraphFlow's built-in LLM utilities
to create agents that can call OpenAI, Anthropic, or Ollama models.

Setup:
1. For OpenAI: Set OPENAI_API_KEY environment variable
2. For Anthropic: Set ANTHROPIC_API_KEY environment variable  
3. For Ollama: Start ollama server (ollama serve) and ensure models are installed

Run: python 08-llm-integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, configure_llm, call_llm, get_llm_config
from typing import TypedDict

class ChatState(TypedDict):
    user_input: str
    conversation: list
    llm_response: str
    should_continue: bool

def setup_llm_node(state: ChatState) -> Command:
    """
    Configure the LLM provider based on available credentials.
    This shows auto-detection of available providers.
    """
    print("🔧 Configuring LLM provider...")

    # Try to configure LLM based on environment
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
            # Test if Ollama is actually running
            call_llm("test")
            print("✅ Using Ollama (local)")
        except Exception:
            print("❌ No LLM provider available!")
            print("Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama")
            return Command(
                update={"should_continue": False},
                goto=END
            )

    # Show current configuration
    config = get_llm_config()
    print(f"   Provider: {config['provider']}")
    print(f"   Model: {config['model']}")
    print()

    return Command(
        update={"should_continue": True},
        goto="get_input"
    )

def get_user_input_node(state: ChatState) -> Command:
    """Get input from the user"""
    if state.get("conversation") is None:
        print("🤖 Welcome to GraphFlow LLM Chat!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-" * 50)
        conversation = []
    else:
        conversation = state["conversation"]

    user_input = input("\n💬 You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'bye', '']:
        return Command(
            update={
                "user_input": user_input,
                "should_continue": False
            },
            goto="goodbye_node"
        )

    # Add user message to conversation
    conversation.append({"role": "user", "content": user_input})

    return Command(
        update={
            "user_input": user_input,
            "conversation": conversation,
            "should_continue": True
        },
        goto="call_llm_node"
    )

def call_llm_node(state: ChatState) -> Command:
    """Call the configured LLM"""
    print("🧠 Thinking...")

    try:
        # Use the conversation history for context
        response = call_llm(state["conversation"])

        # Add assistant response to conversation
        conversation = state["conversation"] + [{"role": "assistant", "content": response}]

        return Command(
            update={
                "llm_response": response,
                "conversation": conversation
            },
            goto="display_response_node"
        )

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        return Command(
            update={"llm_response": error_msg},
            goto="display_response_node"
        )

def display_response_node(state: ChatState) -> Command:
    """Display the LLM response"""
    response = state["llm_response"]
    print(f"🤖 Assistant: {response}")

    return Command(
        update={},
        goto="get_input"
    )

def goodbye_node(state: ChatState) -> Command:
    """Say goodbye"""
    print("\n👋 Thanks for chatting! Goodbye!")

    # Show conversation summary
    if state.get("conversation"):
        print(f"\n📊 Conversation Summary:")
        print(f"   Total messages: {len(state['conversation'])}")
        user_messages = [msg for msg in state['conversation'] if msg['role'] == 'user']
        print(f"   Your messages: {len(user_messages)}")

    return Command(update={}, goto=END)

def main():
    """Run the LLM chat example"""
    print("🚀 GraphFlow LLM Integration Example")
    print("=" * 50)

    # Create the graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("setup_llm", setup_llm_node)
    graph.add_node("get_input", get_user_input_node)
    graph.add_node("call_llm_node", call_llm_node)
    graph.add_node("display_response_node", display_response_node)
    graph.add_node("goodbye_node", goodbye_node)

    # Set entry point
    graph.set_entry_point("setup_llm")

    # Compile and run
    app = graph.compile()

    # Initialize state
    initial_state = {
        "user_input": "",
        "conversation": [],
        "llm_response": "",
        "should_continue": True
    }

    try:
        final_state = app.invoke(initial_state)
        print(f"\n✅ Chat session completed!")

    except KeyboardInterrupt:
        print(f"\n\n👋 Chat interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def demo_provider_switching():
    """Demo showing how to switch between different LLM providers"""
    print("\n🔄 LLM Provider Switching Demo")
    print("=" * 40)

    providers = [
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("ollama", "llama2")
    ]

    test_prompt = "What is 2+2? Answer in exactly 3 words."

    for provider, model in providers:
        print(f"\n🧪 Testing {provider} with {model}...")

        try:
            configure_llm(provider, model=model)
            config = get_llm_config()

            if not config["has_api_key"] and provider != "ollama":
                print(f"   ⏭️  Skipping {provider} (no API key)")
                continue

            response = call_llm(test_prompt)
            print(f"   ✅ Response: {response}")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")

if __name__ == "__main__":
    # Show available options
    print("Choose an option:")
    print("1. Interactive LLM Chat (default)")
    print("2. Provider Switching Demo")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        demo_provider_switching()
    else:
        main()