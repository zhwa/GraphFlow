# GraphFlow LLM Integration Guide

GraphFlow now includes built-in LLM utilities that support multiple providers: **OpenAI**, **Anthropic**, and **Ollama** (local).

## 🚀 Quick Start

```python
from graphflow import StateGraph, Command, call_llm, configure_llm

# Configure your LLM provider
configure_llm("openai", api_key="your-key", model="gpt-4")

# Use in your nodes
def my_llm_node(state):
    response = call_llm("What is the meaning of life?")
    return Command(update={"response": response}, goto="next")
```

## 🔧 Supported Providers

### OpenAI (GPT Models)
```python
configure_llm(
    provider="openai",
    api_key="your-openai-key",  # or set OPENAI_API_KEY env var
    model="gpt-4",
    temperature=0.7
)
```

### Anthropic (Claude Models)
```python
configure_llm(
    provider="anthropic", 
    api_key="your-anthropic-key",  # or set ANTHROPIC_API_KEY env var
    model="claude-3-sonnet-20240229",
    temperature=0.7
)
```

### Ollama (Local Models)
```python
configure_llm(
    provider="ollama",
    base_url="http://localhost:11434",  # default
    model="llama2",  # or any installed model
    temperature=0.7
)
```

### Custom OpenAI-Compatible APIs
```python
configure_llm(
    provider="custom",
    base_url="https://your-api-endpoint.com/v1",
    api_key="your-key",
    model="your-model"
)
```

## 📝 Usage Patterns

### Simple Question-Answer
```python
from graphflow import ask_llm

def qa_node(state):
    answer = ask_llm(state["user_question"])
    return Command(update={"answer": answer}, goto="respond")
```

### Multi-turn Conversation
```python
from graphflow import chat_with_llm

def chat_node(state):
    # state["messages"] is list of {"role": "user/assistant", "content": "..."}
    response = chat_with_llm(state["messages"])
    
    # Add response to conversation
    messages = state["messages"] + [{"role": "assistant", "content": response}]
    return Command(update={"messages": messages}, goto="next")
```

### With Custom Parameters
```python
def creative_node(state):
    response = call_llm(
        prompt="Write a creative story",
        temperature=0.9,  # More creative
        max_tokens=500
    )
    return Command(update={"story": response}, goto="next")
```

### Async Usage
```python
from graphflow import call_llm_async

async def async_llm_node(state):
    response = await call_llm_async(state["prompt"])
    return Command(update={"response": response}, goto="next")
```

## 🛠️ Configuration Management

### Check Current Configuration
```python
from graphflow import get_llm_config

config = get_llm_config()
print(f"Provider: {config['provider']}")
print(f"Model: {config['model']}")
print(f"Has API Key: {config['has_api_key']}")
```

### Auto-Configuration
GraphFlow automatically tries to configure itself on import:

1. **OPENAI_API_KEY** environment variable → Uses OpenAI
2. **ANTHROPIC_API_KEY** environment variable → Uses Anthropic  
3. **Ollama running locally** → Uses Ollama

### Environment Variables
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Ollama (no key needed, just start the server)
ollama serve
```

## 🎯 Complete Example

See `examples/08-llm-integration.py` for a full working example that:
- Auto-detects available LLM providers
- Creates an interactive chat interface
- Handles conversation history
- Shows error handling
- Demonstrates provider switching

## 🔍 Error Handling

```python
def robust_llm_node(state):
    try:
        response = call_llm(state["prompt"])
        return Command(update={"response": response}, goto="success")
    except Exception as e:
        error_msg = f"LLM call failed: {str(e)}"
        return Command(update={"error": error_msg}, goto="error_handler")
```

## 📦 Dependencies

The LLM utilities require additional packages depending on the provider:

```bash
# For OpenAI
pip install openai

# For Anthropic  
pip install anthropic

# For Ollama (no additional packages needed)
# Just install and run Ollama: https://ollama.ai

# For all providers
pip install openai anthropic requests
```

## 🚀 Getting Started

1. **Install dependencies** for your chosen provider
2. **Set environment variables** or call `configure_llm()`
3. **Use `call_llm()`** in your GraphFlow nodes
4. **Run the example**: `python examples/08-llm-integration.py`

The LLM integration makes GraphFlow a complete agent framework with both graph routing intelligence and LLM capabilities! 🎉
