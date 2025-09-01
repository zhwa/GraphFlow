# GraphFlow Tutorial: Your First Agent Graph

This tutorial will walk you through building your first GraphFlow application step by step. By the end, you'll have a working conversational agent that can handle different types of user input.

## Prerequisites

- Python 3.8+
- Basic understanding of Python functions and dictionaries
- Familiarity with type hints (helpful but not required)

## Setup

First, make sure GraphFlow is working:

```bash
# Test the installation
python test_graphflow.py

# Run the quick start
python quick_start.py
```

## Step 1: Understanding State

In GraphFlow, everything revolves around **state** - a shared dictionary that nodes read from and write to.

```python
from typing import TypedDict, List

class ConversationState(TypedDict):
    messages: List[str]          # Conversation history
    user_input: str             # Current user message
    intent: str                 # Detected intent
    confidence: float           # Confidence score
    response: str               # Generated response
```

**Key Concepts:**
- State is typed using `TypedDict` for safety
- All nodes receive the current state
- Nodes return updates to merge into state
- Lists are automatically concatenated by default

## Step 2: Creating Your First Node

Nodes are just Python functions that take state and return updates:

```python
def detect_intent(state: ConversationState) -> dict:
    """Analyze user input to detect intent."""
    user_msg = state["user_input"].lower()
    
    if any(word in user_msg for word in ["help", "assist", "support"]):
        intent = "help_request"
        confidence = 0.9
    elif any(word in user_msg for word in ["hello", "hi", "hey"]):
        intent = "greeting"
        confidence = 0.8
    elif "?" in user_msg:
        intent = "question"
        confidence = 0.7
    else:
        intent = "general"
        confidence = 0.5
    
    return {
        "intent": intent,
        "confidence": confidence
    }
```

**What's happening:**
1. Function receives the complete state
2. Analyzes the `user_input` field
3. Returns a dictionary with updates
4. GraphFlow merges updates into state

## Step 3: Adding Conditional Routing

Use conditional edges to route based on state:

```python
def route_by_intent(state: ConversationState) -> str:
    """Route to appropriate handler based on detected intent."""
    intent = state["intent"]
    confidence = state["confidence"]
    
    # Low confidence - ask for clarification
    if confidence < 0.6:
        return "clarification_handler"
    
    # Route by intent
    if intent == "greeting":
        return "greeting_handler"
    elif intent == "help_request":
        return "help_handler"
    elif intent == "question":
        return "qa_handler"
    else:
        return "general_handler"
```

## Step 4: Creating Handler Nodes

Now create specific handlers for each intent:

```python
def greeting_handler(state: ConversationState) -> dict:
    """Handle greeting messages."""
    responses = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! How are you doing?"
    ]
    import random
    response = random.choice(responses)
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def help_handler(state: ConversationState) -> dict:
    """Handle help requests."""
    response = "I'm here to help! You can ask me questions or just chat. What do you need assistance with?"
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def qa_handler(state: ConversationState) -> dict:
    """Handle questions."""
    response = f"That's an interesting question: '{state['user_input']}'. Let me think about that..."
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def clarification_handler(state: ConversationState) -> dict:
    """Handle unclear input."""
    response = "I'm not sure I understand. Could you please rephrase that or ask me something specific?"
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def general_handler(state: ConversationState) -> dict:
    """Handle general conversation."""
    response = f"You said: '{state['user_input']}'. That's interesting! Tell me more."
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }
```

## Step 5: Building the Graph

Now assemble everything into a graph:

```python
from graphflow import StateGraph

def build_conversation_graph():
    """Build and return the conversation graph."""
    
    # Create the graph
    graph = StateGraph(ConversationState)
    
    # Add all nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("greeting_handler", greeting_handler)
    graph.add_node("help_handler", help_handler)
    graph.add_node("qa_handler", qa_handler)
    graph.add_node("clarification_handler", clarification_handler)
    graph.add_node("general_handler", general_handler)
    
    # Set up routing
    graph.add_conditional_edges("detect_intent", route_by_intent)
    
    # Set entry point
    graph.set_entry_point("detect_intent")
    
    return graph.compile()
```

## Step 6: Testing Your Graph

Create a simple test function:

```python
def test_conversation():
    """Test the conversation graph with different inputs."""
    
    graph = build_conversation_graph()
    
    test_inputs = [
        "Hello there!",
        "Can you help me?",
        "What is the weather like?",
        "asdf jkl;",  # Unclear input
        "I love programming"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        print("-" * 40)
        
        # Run the graph
        result = graph.invoke({
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "confidence": 0.0,
            "response": ""
        })
        
        print(f"Detected Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Bot Response: {result['response']}")

if __name__ == "__main__":
    test_conversation()
```

## Step 7: Adding Command Objects

For more complex control flow, use Command objects:

```python
from graphflow import Command, END

def smart_handler(state: ConversationState) -> Command:
    """Handler that can decide whether to continue or end."""
    
    user_msg = state["user_input"].lower()
    
    # Check for goodbye
    if any(word in user_msg for word in ["bye", "goodbye", "exit", "quit"]):
        return Command(
            update={
                "response": "Goodbye! Have a great day!",
                "messages": state["messages"] + [f"User: {state['user_input']}", "Bot: Goodbye! Have a great day!"]
            },
            goto=END  # End the conversation
        )
    
    # Check for follow-up questions
    elif "more" in user_msg or "tell me" in user_msg:
        return Command(
            update={
                "response": "What would you like to know more about?",
                "messages": state["messages"] + [f"User: {state['user_input']}", "Bot: What would you like to know more about?"]
            },
            goto="detect_intent"  # Loop back for more conversation
        )
    
    # Default response
    else:
        return Command(
            update={
                "response": "I see. Anything else I can help with?",
                "messages": state["messages"] + [f"User: {state['user_input']}", "Bot: I see. Anything else I can help with?"]
            },
            goto=END  # End after response
        )
```

## Complete Example

Here's the complete tutorial code:

```python
# tutorial_example.py
from graphflow import StateGraph, Command, END
from typing import TypedDict, List
import random

class ConversationState(TypedDict):
    messages: List[str]
    user_input: str
    intent: str
    confidence: float
    response: str

def detect_intent(state: ConversationState) -> dict:
    user_msg = state["user_input"].lower()
    
    if any(word in user_msg for word in ["help", "assist", "support"]):
        intent, confidence = "help_request", 0.9
    elif any(word in user_msg for word in ["hello", "hi", "hey"]):
        intent, confidence = "greeting", 0.8
    elif "?" in user_msg:
        intent, confidence = "question", 0.7
    elif any(word in user_msg for word in ["bye", "goodbye"]):
        intent, confidence = "goodbye", 0.9
    else:
        intent, confidence = "general", 0.5
    
    return {"intent": intent, "confidence": confidence}

def route_by_intent(state: ConversationState) -> str:
    intent = state["intent"]
    confidence = state["confidence"]
    
    if confidence < 0.6:
        return "clarification_handler"
    
    handlers = {
        "greeting": "greeting_handler",
        "help_request": "help_handler", 
        "question": "qa_handler",
        "goodbye": "goodbye_handler",
        "general": "general_handler"
    }
    
    return handlers.get(intent, "general_handler")

def greeting_handler(state: ConversationState) -> dict:
    responses = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! How are you doing?"
    ]
    response = random.choice(responses)
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def help_handler(state: ConversationState) -> dict:
    response = "I'm here to help! You can ask me questions or just chat. What do you need assistance with?"
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def qa_handler(state: ConversationState) -> dict:
    response = f"That's an interesting question: '{state['user_input']}'. Let me think about that..."
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def goodbye_handler(state: ConversationState) -> Command:
    response = "Goodbye! Thanks for chatting with me!"
    
    return Command(
        update={
            "response": response,
            "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
        },
        goto=END
    )

def clarification_handler(state: ConversationState) -> dict:
    response = "I'm not sure I understand. Could you please rephrase that?"
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def general_handler(state: ConversationState) -> dict:
    response = f"You said: '{state['user_input']}'. That's interesting! Tell me more."
    
    return {
        "response": response,
        "messages": state["messages"] + [f"User: {state['user_input']}", f"Bot: {response}"]
    }

def build_conversation_graph():
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("greeting_handler", greeting_handler)
    graph.add_node("help_handler", help_handler)
    graph.add_node("qa_handler", qa_handler)
    graph.add_node("goodbye_handler", goodbye_handler)
    graph.add_node("clarification_handler", clarification_handler)
    graph.add_node("general_handler", general_handler)
    
    # Add routing
    graph.add_conditional_edges("detect_intent", route_by_intent)
    graph.set_entry_point("detect_intent")
    
    return graph.compile()

def main():
    graph = build_conversation_graph()
    
    test_inputs = [
        "Hello there!",
        "Can you help me?", 
        "What is the weather like?",
        "asdf jkl;",
        "I love programming",
        "Goodbye!"
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"Testing: {user_input}")
        print('='*50)
        
        result = graph.invoke({
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "confidence": 0.0,
            "response": ""
        })
        
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Response: {result['response']}")
        print("\nFull conversation:")
        for i, msg in enumerate(result['messages'], 1):
            print(f"  {i}. {msg}")

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you've built your first GraphFlow application, try:

1. **Add more intents** - sentiment analysis, topic detection
2. **Implement memory** - remember previous conversations
3. **Add external tools** - web search, database queries
4. **Create loops** - multi-turn conversations
5. **Add error handling** - graceful failure management

## Key Takeaways

1. **State is central** - Everything flows through shared state
2. **Nodes are simple** - Just functions that read and update state
3. **Routing is flexible** - Conditional edges enable complex flows
4. **Commands add power** - Combine updates with control flow
5. **Composition works** - Build complex agents from simple parts

Continue with [Usage Patterns](../examples/02-core-patterns.py) to see more advanced techniques!
