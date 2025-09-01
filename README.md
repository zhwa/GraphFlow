# GraphFlow

**A Lightweight State-Based Agent Framework**

Welcome to GraphFlow - a simple, powerful agent framework that combines state-based routing with minimalist design. Built for developers who want LangGraph-like functionality without the complexity.

## 🚀 Quick Start

Get running in under 2 minutes:

```bash
cd graphflow
pip install -r requirements.txt
python examples/00-quick-start.py
```

## 📖 Documentation

Our documentation is organized in a flat, numbered structure for easy navigation:

### **Getting Started (00-03)**
- **[📋 Project Summary](docs/00-project-summary.md)** - What GraphFlow is and why it exists
- **[🚀 Quick Start Guide](docs/01-quick-start.md)** - Get running in 5 minutes  
- **[💿 Installation](docs/02-installation.md)** - Setup and environment
- **[🎯 First Example](docs/03-first-example.md)** - Build your first agent

### **Core Concepts (04-07)**
- **[🏗️ Architecture](docs/04-architecture.md)** - How GraphFlow works
- **[📊 State Management](docs/05-state-management.md)** - TypedDict schemas and state flow
- **[📚 Comprehensive Guide](docs/06-comprehensive-guide.md)** - Complete feature overview
- **[🗂️ Documentation Index](docs/07-documentation-index.md)** - Navigation hub

### **Reference (08-09)**
- **[🆚 Framework Comparison](docs/08-comparison.md)** - GraphFlow vs LangGraph vs others
- **[📖 API Reference](docs/09-api-reference.md)** - Complete API documentation

### **Examples**
- **[💡 Working Examples](examples/)** - 8 progressive examples from basic to advanced

---

## 🌟 Quick Preview

Here's what GraphFlow looks like in action:

```python
from graphflow import StateGraph, Command, END
from typing import TypedDict

class ChatState(TypedDict):
    messages: list
    user_input: str
    should_continue: bool

def process_message(state: ChatState) -> Command:
    user_input = state["user_input"].lower()
    
    if "goodbye" in user_input:
        response = "Goodbye! Thanks for chatting."
        return Command(
            update={
                "messages": state["messages"] + [f"Bot: {response}"],
                "should_continue": False
            },
            goto=END
        )
    
    response = f"I heard you say: {state['user_input']}"
    return Command(
        update={
            "messages": state["messages"] + [f"Bot: {response}"],
            "should_continue": True
        },
        goto="wait_for_input"
    )

# Build the conversational agent
graph = StateGraph(ChatState)
graph.add_node("process", process_message)
graph.add_node("wait_for_input", lambda state: {"ready": True})
graph.set_entry_point("process")

app = graph.compile()
```

**👆 This creates a conversational agent in ~25 lines of code with zero external dependencies!**

Test your installation and see more examples:
```bash
python examples/00-quick-start.py          # Gentle introduction (✓ WORKING!)
python examples/01-test-and-examples.py    # Comprehensive test suite (✓ WORKING!)
```

---

## 🎯 Choose Your Learning Path

**🔰 New to agent frameworks?**  
Start with [Quick Start Guide](docs/01-quick-start.md)

**💻 Experienced developer?**  
Jump to [Architecture](docs/04-architecture.md)

**🔍 Need specific examples?**  
Browse the [Examples Directory](examples/)

**📚 Want complete documentation?**  
Check the [Documentation Index](docs/07-documentation-index.md)

**❓ Have questions?**  
Check the [API Reference](docs/09-api-reference.md)

---

## 💡 What Makes GraphFlow Special

- **🪶 Lightweight**: Simple, focused codebase
- **🚫 Zero Dependencies**: Only Python standard library  
- **🔄 LangGraph Compatible**: Familiar API for easy migration
- **⚡ Immediate Use**: No setup required - examples work instantly
- **🧪 Test Friendly**: Built for easy testing and validation
- **📖 Well Documented**: Complete guides and working examples

---

**Ready to build something amazing? Start with [Quick Start Guide](docs/01-quick-start.md)!** 🚀
