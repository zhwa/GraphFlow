"""
GraphFlow Example: Hello World QA with Real LLM
Ported from: PocketFlow cookbook/pocketflow-hello-world

This example demonstrates the simplest possible GraphFlow application:
a single-node graph that answers questions using a real LLM.

Setup:
1. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama
2. Run: python 03-hello-world-qa.py

Original PocketFlow pattern:
- AnswerNode with prep/exec/post methods
- Shared state dictionary
- Flow with single node

GraphFlow adaptation:
- Simple function-based node with real LLM calls
- TypedDict state schema
- StateGraph with single node
"""

import sys
import os
from typing import TypedDict

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, call_llm, configure_llm, get_llm_config

# State schema - defines what data flows through the graph
class QAState(TypedDict):
    question: str
    answer: str

def setup_llm() -> bool:
    """Configure LLM provider and return True if successful."""
    if os.environ.get("OPENAI_API_KEY"):
        configure_llm("openai", model="gpt-4")
        print("✅ Using OpenAI GPT-4")
        return True
    elif os.environ.get("ANTHROPIC_API_KEY"):
        configure_llm("anthropic", model="claude-3-sonnet-20240229")
        print("✅ Using Anthropic Claude")
        return True
    else:
        try:
            configure_llm("ollama", model="llama2")
            call_llm("test")  # Test if Ollama is working
            print("✅ Using Ollama (local)")
            return True
        except Exception:
            print("❌ No LLM provider available!")
            print("\nTo use this example, you need:")
            print("1. Set OPENAI_API_KEY environment variable, OR")
            print("2. Set ANTHROPIC_API_KEY environment variable, OR")
            print("3. Start Ollama server: ollama serve")
            return False

def answer_question(state: QAState) -> dict:
    """
    Process a question and return an answer using real LLM.
    
    In PocketFlow, this was split into prep/exec/post:
    - prep: extract question from shared state
    - exec: call LLM with question
    - post: store answer back in shared state
    
    In GraphFlow, we do it all in one function with real LLM calls.
    """
    question = state["question"]
    
    try:
        # Call real LLM with a well-structured prompt
        prompt = f"""You are a helpful AI assistant. Please answer the following question clearly and concisely:

Question: {question}

Please provide a helpful, informative answer."""
        
        answer = call_llm(prompt)
        
    except Exception as e:
        answer = f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    # Return state updates
    return {"answer": answer}

def build_qa_graph():
    """Build the QA graph - equivalent to PocketFlow's Flow(start=answer_node)"""
    graph = StateGraph(QAState)
    
    # Add our single node
    graph.add_node("answer", answer_question)
    
    # Set it as the entry point
    graph.set_entry_point("answer")
    
    # Compile to runnable form
    return graph.compile()

def main():
    """Main function with real LLM integration."""
    print("🤖 GraphFlow Hello World QA with Real LLM")
    print("=" * 50)
    
    # Setup LLM first
    if not setup_llm():
        return
    
    config = get_llm_config()
    print(f"Provider: {config['provider']} | Model: {config['model']}")
    print("-" * 50)
    
    # Build the graph
    qa_app = build_qa_graph()
    
    # Test questions
    test_questions = [
        "What is the meaning of life?",
        "How does artificial intelligence work?", 
        "What is quantum physics?",
        "Explain black holes in simple terms"
    ]
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        print("🧠 Thinking...")
        
        # Run the graph - equivalent to qa_flow.run(shared)
        result = qa_app.invoke({
            "question": question,
            "answer": ""  # Initial empty answer
        })
        
        print(f"🤖 Answer: {result['answer']}")

def interactive_mode():
    """Interactive mode for testing real LLM"""
    print("\n🎮 Interactive QA Mode")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    if not setup_llm():
        return
        
    qa_app = build_qa_graph()
    
    while True:
        question = input("\n🤔 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not question:
            print("Please enter a question.")
            continue
            
        print("🧠 Thinking...")
        
        try:
            result = qa_app.invoke({
                "question": question,
                "answer": ""
            })
            print(f"🤖 Answer: {result['answer']}")
        except Exception as e:
            print(f"❌ Error: {e}")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        result = qa_app.invoke({
            "question": question,
            "answer": ""
        })
        
        print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    # Run basic demo
    main()
    
    # Uncomment for interactive mode
    # interactive_mode()

"""
Key Differences from PocketFlow:

1. State Schema:
   - PocketFlow: Untyped dictionary
   - GraphFlow: TypedDict with explicit fields

2. Node Definition:
   - PocketFlow: Class with prep/exec/post methods
   - GraphFlow: Single function returning state updates

3. Flow Construction:
   - PocketFlow: Flow(start=node)
   - GraphFlow: StateGraph with add_node/set_entry_point/compile

4. Execution:
   - PocketFlow: flow.run(shared) modifies shared dict
   - GraphFlow: app.invoke(state) returns new state

5. Type Safety:
   - PocketFlow: Runtime errors for missing keys
   - GraphFlow: IDE autocomplete and type checking

Benefits of GraphFlow approach:
- Better IDE support with autocomplete
- Type safety catches errors early
- Immutable state updates (no side effects)
- Easier to test individual nodes
- More functional programming style
"""
