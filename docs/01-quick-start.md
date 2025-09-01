# Chapter 1: Quick Start
**Get GraphFlow running in 5 minutes**

Welcome to GraphFlow! This chapter will get you up and running quickly with a working example, so you can see what GraphFlow can do before diving into the details.

## What You'll Build

By the end of this chapter, you'll have a working conversational agent that can:
- Process user messages
- Remember conversation history
- Handle goodbye messages gracefully
- Route between different conversation states

## Prerequisites

- Python 3.8 or higher
- Basic understanding of Python functions and dictionaries
- 5 minutes of your time!

## Step 1: Get GraphFlow

GraphFlow has zero external dependencies - it's just a single Python file!

```bash
# Download or clone GraphFlow
git clone <your-repo-url>
cd graphflow

# That's it! No pip install needed.
```

## Step 2: Your First Agent

Create a new file called `my_first_agent.py`:

```python
from graphflow import StateGraph, Command, END
from typing import TypedDict, List

# Step 1: Define what data your agent will work with
class ConversationState(TypedDict):
    messages: List[str]          # Chat history
    user_input: str             # What the user just said
    turn_count: int             # How many turns we've had
    should_continue: bool       # Whether to keep chatting

# Step 2: Create functions that process the state
def greet_user(state: ConversationState) -> dict:
    """Start the conversation with a greeting."""
    greeting = "Hello! I'm your GraphFlow assistant. How can I help you today?"
    return {
        "messages": [f"Assistant: {greeting}"],
        "turn_count": 1
    }

def process_user_message(state: ConversationState) -> Command:
    """Process what the user said and decide what to do next."""
    user_msg = state["user_input"].lower().strip()
    current_messages = state["messages"]
    turn = state["turn_count"]
    
    # Add user message to history
    updated_messages = current_messages + [f"User: {state['user_input']}"]
    
    # Check if user wants to end conversation
    if any(word in user_msg for word in ["bye", "goodbye", "quit", "exit"]):
        farewell = "Goodbye! It was nice chatting with you."
        return Command(
            update={
                "messages": updated_messages + [f"Assistant: {farewell}"],
                "should_continue": False,
                "turn_count": turn + 1
            },
            goto=END  # End the conversation
        )
    
    # Generate a response based on the input
    if "hello" in user_msg or "hi" in user_msg:
        response = "Hello there! Nice to meet you."
    elif "how are you" in user_msg:
        response = "I'm doing great, thanks for asking! How are you?"
    elif "?" in user_msg:
        response = f"That's an interesting question: '{state['user_input']}'. Let me think about that..."
    else:
        response = f"I heard you say: '{state['user_input']}'. Tell me more!"
    
    # Continue the conversation
    return Command(
        update={
            "messages": updated_messages + [f"Assistant: {response}"],
            "should_continue": True,
            "turn_count": turn + 1
        },
        goto="wait_for_input"  # Go to waiting state
    )

def wait_for_input(state: ConversationState) -> dict:
    """Wait for the next user input."""
    return {"ready_for_input": True}

# Step 3: Build the conversation graph
def build_conversation_agent():
    # Create a new graph with our state schema
    graph = StateGraph(ConversationState)
    
    # Add our processing functions as nodes
    graph.add_node("greet", greet_user)
    graph.add_node("process", process_user_message)
    graph.add_node("wait_for_input", wait_for_input)
    
    # Define the flow between nodes
    graph.add_edge("greet", "wait_for_input")      # After greeting, wait for input
    graph.add_edge("wait_for_input", "process")    # After waiting, process input
    
    # Start the conversation with a greeting
    graph.set_entry_point("greet")
    
    # Compile the graph into a runnable application
    return graph.compile()

# Step 4: Test your agent
def main():
    print("Building your first GraphFlow agent...")
    agent = build_conversation_agent()
    
    # Test conversation scenarios
    test_inputs = [
        "Hello there!",
        "How are you doing?", 
        "What's the weather like?",
        "Goodbye!"
    ]
    
    # Start with initial state
    state = {
        "messages": [],
        "user_input": "",
        "turn_count": 0,
        "should_continue": True
    }
    
    # Run the greeting
    result = agent.invoke(state)
    print("\\n=== Conversation Started ===")
    for message in result["messages"]:
        print(message)
    
    # Simulate conversation turns
    for user_input in test_inputs:
        if not result.get("should_continue", True):
            break
            
        # User speaks
        result["user_input"] = user_input
        
        # Agent processes and responds
        result = agent.invoke(result)
        
        # Show the latest messages
        if len(result["messages"]) > 0:
            print(result["messages"][-2])  # User message
            print(result["messages"][-1])  # Assistant response
            print()  # Blank line for readability
    
    print("=== Conversation Ended ===")
    print(f"Total turns: {result['turn_count']}")

if __name__ == "__main__":
    main()
```

## Step 3: Run Your Agent

```bash
python my_first_agent.py
```

You should see output like:

```
Building your first GraphFlow agent...

=== Conversation Started ===
Assistant: Hello! I'm your GraphFlow assistant. How can I help you today?
User: Hello there!
Assistant: Hello there! Nice to meet you.

User: How are you doing?
Assistant: I'm doing great, thanks for asking! How are you?

User: What's the weather like?
Assistant: That's an interesting question: 'What's the weather like?'. Let me think about that...

User: Goodbye!
Assistant: Goodbye! It was nice chatting with you.

=== Conversation Ended ===
Total turns: 5
```

## What Just Happened?

Congratulations! You just built your first GraphFlow agent. Let's break down what happened:

### 1. State Definition
```python
class ConversationState(TypedDict):
    messages: List[str]
    user_input: str
    turn_count: int
    should_continue: bool
```
This defines the "memory" of your agent - what data it keeps track of during conversations.

### 2. Processing Functions
Each function receives the current state and returns updates:
- `greet_user()` - Starts the conversation
- `process_user_message()` - Handles user input and generates responses
- `wait_for_input()` - Manages the waiting state

### 3. Graph Construction
```python
graph = StateGraph(ConversationState)
graph.add_node("greet", greet_user)
graph.add_node("process", process_user_message)
# ... connect them with edges
```
This builds the workflow structure.

### 4. Smart Routing with Commands
```python
return Command(
    update={"messages": updated_messages, ...},
    goto="wait_for_input"  # or END
)
```
Commands let you update state AND control where to go next in a single operation.

## Key Concepts You've Learned

- **State-Centric Design**: Everything revolves around updating shared state
- **Node Functions**: Simple Python functions that process state
- **Commands**: Combine state updates with routing decisions
- **Graph Structure**: Nodes connected by edges define your workflow
- **Type Safety**: TypedDict gives you autocomplete and error checking

## What's Next?

You've successfully built a working agent! Here are your next steps:

- **[Chapter 2: Installation](02-installation.md)** - Set up a proper development environment
- **[Chapter 3: First Example](03-first-example.md)** - Build something more complex
- **[Chapter 4: Architecture](04-architecture.md)** - Understand how GraphFlow works under the hood

## Try This Yourself

Before moving on, try modifying your agent:

1. **Add new responses** - Make it respond to different keywords
2. **Track more state** - Add a mood or topic tracker
3. **Add more nodes** - Create specialized handlers for different conversation types
4. **Experiment with routing** - Use conditional edges to create branching conversations

```python
# Example: Add mood tracking
class EnhancedState(ConversationState):
    user_mood: str  # "happy", "sad", "neutral"
    topics_discussed: List[str]

def detect_mood(state: EnhancedState) -> dict:
    user_msg = state["user_input"].lower()
    
    if any(word in user_msg for word in ["great", "awesome", "happy"]):
        mood = "happy"
    elif any(word in user_msg for word in ["sad", "bad", "terrible"]):
        mood = "sad"
    else:
        mood = "neutral"
    
    return {"user_mood": mood}
```

The beauty of GraphFlow is its simplicity - you can start small and gradually add complexity as you learn!

---

**Next: [Chapter 2: Installation →](02-installation.md)**
