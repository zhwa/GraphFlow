"""
GraphFlow Example: Simple Q&A System

This example demonstrates a simple question-answering system that:
- Processes user questions
- Attempts to answer using an LLM 
- Falls back gracefully when LLM is unavailable
- Shows basic error handling patterns

Key GraphFlow features demonstrated:
- Single node processing
- LLM integration with fallbacks
- Simple state management
- Clean error handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph
from llm_utils import call_llm

def answer_question(state):
    """
    Process a user question and provide an answer.
    """
    question = state.get("question", "")
    
    if not question:
        return {"answer": "Please provide a question to answer."}
    
    # Try to get LLM response
    try:
        prompt = f"Please answer this question clearly and concisely: {question}"
        response = call_llm(prompt)
        return {"answer": response.strip()}
    except Exception as e:
        print(f"LLM unavailable: {e}")
    
    # Fallback responses for common questions
    fallback_answers = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What would you like to know?",
        "how are you": "I'm doing well, thank you for asking!",
        "what is your name": "I'm a GraphFlow-powered Q&A assistant.",
        "goodbye": "Goodbye! Have a great day!",
        "bye": "Bye! Come back anytime!"
    }
    
    # Simple keyword matching for fallback
    question_lower = question.lower().strip()
    for key, answer in fallback_answers.items():
        if key in question_lower:
            return {"answer": answer}
    
    # Default fallback
    return {
        "answer": "I'm sorry, I don't have an answer for that question. "
                 "(LLM service is currently unavailable)"
    }

def main():
    """Demonstrate the simple Q&A system."""
    print("GraphFlow Simple Q&A System")
    print("========================================")
    
    # Build the graph
    graph = StateGraph()
    graph.add_node("answer", answer_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", "__end__")
    
    # Compile the application
    app = graph.compile()
    
    # Test questions
    test_questions = [
        "Hello there!",
        "What is the capital of France?",
        "How are you?",
        "What is machine learning?",
        "Goodbye!"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        try:
            result = app.invoke({"question": question})
            print(f"Answer: {result.get('answer', 'No answer provided')}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
