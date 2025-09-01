# Chapter 2: Installation & Setup
**Setting up your GraphFlow development environment**

This chapter covers different ways to install and set up GraphFlow, from simple single-file usage to full development environments.

## Prerequisites

- **Python 3.8+** (3.9+ recommended)
- Basic familiarity with Python virtual environments
- A text editor or IDE (VS Code, PyCharm, etc.)

## Installation Options

### Option 1: Simple Download (Recommended for Beginners)

GraphFlow is designed to be dependency-free and portable:

```bash
# Download just the files you need
curl -O https://raw.githubusercontent.com/your-repo/graphflow/main/graphflow.py

# Or clone the entire repository
git clone https://github.com/your-repo/graphflow.git
cd graphflow
```

That's it! GraphFlow has zero external dependencies, so you can start using it immediately:

```python
# test_installation.py
from graphflow import StateGraph
print("GraphFlow installed successfully!")
```

### Option 2: Development Setup

For serious development work, set up a proper environment:

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/graphflow.git
cd graphflow

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Run the setup script (optional but recommended)
python setup.py

# 5. Verify installation
python test_graphflow.py
```

### Option 3: Embedding in Your Project

For production projects, copy GraphFlow into your codebase:

```bash
# Copy the core file to your project
cp path/to/graphflow/graphflow.py your_project/
```

Then import it directly:

```python
# In your project
from .graphflow import StateGraph, Command, END
```

## Project Structure

After installation, your GraphFlow directory should look like:

```
graphflow/
├── graphflow.py           # Core framework (the only required file)
├── examples.py           # Example applications
├── test_graphflow.py     # Test suite
├── quick_start.py        # Quick demonstration
├── setup.py             # Development setup script
├── README.md            # Project overview
└── docs/                # Complete documentation
    ├── 01-quick-start.md
    ├── 02-installation.md
    ├── ...
    ├── examples/        # Example patterns
    └── docs/            # Documentation files (flattened structure)
```

## Verification

### Quick Test

```python
# test_install.py
from graphflow import StateGraph, Command, END
from typing import TypedDict

class TestState(TypedDict):
    value: int

def increment(state: TestState) -> dict:
    return {"value": state["value"] + 1}

# Build and test
graph = StateGraph(TestState)
graph.add_node("inc", increment)
graph.set_entry_point("inc")

app = graph.compile()
result = app.invoke({"value": 0})

assert result["value"] == 1
print("✅ GraphFlow is working correctly!")
```

### Run the Test Suite

```bash
python test_graphflow.py
```

Expected output:
```
test_basic_functionality (__main__.TestGraphFlow) ... ok
test_conditional_routing (__main__.TestGraphFlow) ... ok
test_command_functionality (__main__.TestGraphFlow) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
```

### Try the Examples

```bash
# Quick demonstration
python quick_start.py

# Comprehensive examples
python examples.py
```

## Development Tools

### Recommended IDE Setup

**VS Code Extensions:**
- Python (Microsoft)
- Python Type Hint (Microsoft)
- Python Docstring Generator
- GitLens (for version control)

**VS Code Settings (`.vscode/settings.json`):**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.typeChecking": "basic"
}
```

### Optional Dependencies for Development

While GraphFlow has zero runtime dependencies, these tools are helpful for development:

```bash
# Install development tools (optional)
pip install black           # Code formatting
pip install pylint          # Code linting
pip install mypy            # Type checking
pip install pytest          # Testing framework
```

Create a `requirements-dev.txt`:
```
black>=22.0.0
pylint>=2.15.0
mypy>=0.991
pytest>=7.0.0
```

## Environment Configuration

### Environment Variables

GraphFlow doesn't require any environment variables, but you might want to set these for development:

```bash
# .env file (optional)
PYTHONPATH=.
GRAPHFLOW_DEBUG=true
GRAPHFLOW_LOG_LEVEL=INFO
```

### Development Scripts

Create helper scripts for common tasks:

**`scripts/test.py`:**
```python
#!/usr/bin/env python3
import subprocess
import sys

def run_tests():
    """Run all tests."""
    result = subprocess.run([sys.executable, "test_graphflow.py"])
    return result.returncode == 0

def run_examples():
    """Run example scripts."""
    examples = ["quick_start.py", "examples.py"]
    for example in examples:
        print(f"Running {example}...")
        result = subprocess.run([sys.executable, example])
        if result.returncode != 0:
            print(f"❌ {example} failed")
            return False
        print(f"✅ {example} passed")
    return True

if __name__ == "__main__":
    print("Running GraphFlow test suite...")
    if run_tests() and run_examples():
        print("🎉 All tests and examples passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
```

**`scripts/format.py`:**
```python
#!/usr/bin/env python3
import subprocess

def format_code():
    """Format code with black."""
    subprocess.run(["black", "graphflow.py", "examples.py", "test_graphflow.py"])
    print("✅ Code formatted")

def lint_code():
    """Lint code with pylint."""
    files = ["graphflow.py", "examples.py", "test_graphflow.py"]
    for file in files:
        print(f"Linting {file}...")
        subprocess.run(["pylint", file])

if __name__ == "__main__":
    format_code()
    lint_code()
```

## Platform-Specific Notes

### Windows

```cmd
# Use PowerShell or Command Prompt
git clone https://github.com/your-repo/graphflow.git
cd graphflow
python -m venv venv
venv\Scripts\activate
python test_graphflow.py
```

### macOS

```bash
# Install Python if needed
brew install python@3.9

# Standard setup
git clone https://github.com/your-repo/graphflow.git
cd graphflow
python3 -m venv venv
source venv/bin/activate
python test_graphflow.py
```

### Linux

```bash
# Install Python if needed (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

# Standard setup
git clone https://github.com/your-repo/graphflow.git
cd graphflow
python3 -m venv venv
source venv/bin/activate
python test_graphflow.py
```

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'graphflow'`**
```python
# Solution: Ensure graphflow.py is in your Python path
import sys
sys.path.append('/path/to/graphflow')
from graphflow import StateGraph
```

**Issue: Type hints not working**
```python
# Solution: Ensure you have Python 3.8+ and typing_extensions if needed
from typing import TypedDict  # Python 3.8+
# or
from typing_extensions import TypedDict  # Python 3.7
```

**Issue: Tests failing on Windows**
```python
# Solution: Use raw strings or forward slashes for paths
import os
path = os.path.join("docs", "examples")  # Cross-platform
```

### Validation Script

Create `validate_setup.py`:

```python
#!/usr/bin/env python3
"""Validate GraphFlow installation and setup."""

import sys
import os
import importlib.util

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_graphflow():
    """Check GraphFlow import."""
    try:
        from graphflow import StateGraph, Command, END
        print("✅ GraphFlow imports working")
        return True
    except ImportError as e:
        print(f"❌ GraphFlow import failed: {e}")
        return False

def check_typing():
    """Check typing support."""
    try:
        from typing import TypedDict
        print("✅ Typing support available")
        return True
    except ImportError:
        print("❌ Typing support missing")
        return False

def check_files():
    """Check required files exist."""
    required_files = ["graphflow.py", "examples.py", "test_graphflow.py"]
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    return True

def main():
    """Run all validation checks."""
    print("Validating GraphFlow setup...")
    print("-" * 40)
    
    checks = [
        check_python_version,
        check_graphflow,
        check_typing,
        check_files
    ]
    
    results = [check() for check in checks]
    
    print("-" * 40)
    if all(results):
        print("🎉 Setup validation successful!")
        print("You're ready to start building with GraphFlow!")
        return True
    else:
        print("❌ Setup validation failed")
        print("Please check the issues above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
python validate_setup.py
```

## Next Steps

Now that GraphFlow is installed and verified:

1. **[Chapter 3: First Example](03-first-example.md)** - Build your first real application
2. **[Chapter 4: Architecture](04-architecture.md)** - Understand the design principles
3. **[Example Gallery](../examples/)** - Explore real-world patterns

## Getting Help

If you encounter installation issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Run the validation script
3. Check the [GitHub issues](https://github.com/your-repo/graphflow/issues)
4. Create a new issue with your setup details

---

**Next: [Chapter 3: First Example →](03-first-example.md)**
