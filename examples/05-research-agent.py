"""
GraphFlow Example: Research Agent with Web Search and Real LLM
Ported from: PocketFlow cookbook/pocketflow-agent

This example demonstrates an intelligent agent that:
- Uses real LLM to analyze questions and generate answers
- Performs web searches when needed (simulated)
- Accumulates information across multiple search iterations
- Uses conditional routing for decision making

Setup:
1. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama
2. Run: python 05-research-agent.py

Original PocketFlow pattern:
- Multiple nodes (DecideAction, SearchWeb, AnswerQuestion)
- Conditional routing with string returns ("search", "answer", "decide")
- Loop-back mechanism for iterative searching

GraphFlow adaptation:
- Real LLM integration for intelligent decision making
- State-based knowledge accumulation
- Command objects for flow control
- Cleaner conditional routing
- Type-safe state management
"""

import sys
import os
from typing import TypedDict, List, Dict, Optional
import re

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, call_llm, configure_llm, get_llm_config

# State schema for research agent
class ResearchState(TypedDict):
    question: str                    # Original question
    research_context: List[str]      # Accumulated research findings
    search_queries: List[str]        # Queries that have been searched
    current_analysis: str           # Current analysis of the question
    confidence_level: float         # Confidence in current knowledge (0-1)
    max_searches: int              # Maximum number of searches allowed
    searches_performed: int        # Number of searches performed so far
    final_answer: Optional[str]    # Final answer when ready
    reasoning_steps: List[str]     # Steps taken in reasoning process
    last_decision: str             # Last decision from analysis (SEARCH/ANSWER)
    suggested_queries: List[str]   # Queries suggested by LLM
    answer: str                    # Final generated answer

def setup_llm_node(state: ResearchState) -> Command:
    """Configure LLM provider for the research agent."""
    print("🔧 Setting up LLM provider for research agent...")
    
    if os.environ.get("OPENAI_API_KEY"):
        configure_llm("openai", model="gpt-4")
        print("✅ Using OpenAI GPT-4")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        configure_llm("anthropic", model="claude-3-sonnet-20240229")
        print("✅ Using Anthropic Claude")
    else:
        try:
            configure_llm("ollama", model="llama2")
            call_llm("test")
            print("✅ Using Ollama (local)")
        except Exception:
            print("❌ No LLM provider available!")
            print("\nTo use this example, you need:")
            print("1. Set OPENAI_API_KEY environment variable, OR")
            print("2. Set ANTHROPIC_API_KEY environment variable, OR")
            print("3. Start Ollama server: ollama serve")
            return Command(update={}, goto=END)
    
    config = get_llm_config()
    print(f"   Provider: {config['provider']} | Model: {config['model']}")
    print("-" * 50)
    
    return Command(update={}, goto="analyze_question")

def analyze_question(state: ResearchState) -> Command:
    """
    Use real LLM to analyze the question and decide whether we need more information.
    
    Equivalent to PocketFlow's DecideAction node.
    """
    question = state["question"]
    research_context = state["research_context"]
    searches_performed = state["searches_performed"]
    max_searches = state["max_searches"]
    
    # Add analysis step
    reasoning_steps = state["reasoning_steps"] + [
        f"Analyzing question: '{question}'"
    ]
    
    print(f"🤔 Analyzing question with LLM...")
    
    # Use LLM to analyze the question and decide if more research is needed
    analysis_prompt = f"""You are a research agent. Analyze this question and determine if you need more information to answer it well.

Question: {question}

Current research context:
{chr(10).join(research_context) if research_context else "No research performed yet"}

Searches performed: {searches_performed}/{max_searches}

Based on the question and available context, respond with ONLY one of these options:
- SEARCH: if you need more specific information to answer well
- ANSWER: if you have enough information to provide a good answer

If SEARCH, also suggest 1-3 specific search queries that would help answer the question.
If ANSWER, explain why you have sufficient information.

Format your response as:
DECISION: [SEARCH or ANSWER]
REASONING: [Your reasoning]
SEARCH_QUERIES: [If SEARCH, list queries separated by | ]"""

    try:
        llm_response = call_llm(analysis_prompt)
        
        # Parse LLM response
        decision = "ANSWER"  # default
        search_queries = []
        reasoning = "Default reasoning"
        
        if "DECISION:" in llm_response:
            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("DECISION:"):
                    decision = line.replace("DECISION:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("SEARCH_QUERIES:") and "SEARCH" in decision:
                    queries_text = line.replace("SEARCH_QUERIES:", "").strip()
                    search_queries = [q.strip() for q in queries_text.split('|') if q.strip()]
        
        reasoning_steps.append(f"LLM analysis: {reasoning}")
        
        # Store the decision for the routing function to use
        return Command(
            update={
                "reasoning_steps": reasoning_steps,
                "current_analysis": reasoning,
                "last_decision": decision,  # Store for router
                "suggested_queries": search_queries
            }
            # No goto - let conditional routing decide
        )
            
    except Exception as e:
        print(f"⚠️  LLM analysis failed: {e}")
        # Fallback to simple logic
        decision = "SEARCH" if searches_performed < max_searches and len(research_context) < 2 else "ANSWER"
        return Command(
            update={
                "reasoning_steps": reasoning_steps + ["Fallback: using simple heuristics"],
                "current_analysis": "LLM analysis failed, using fallback logic",
                "last_decision": decision
            }
        )

def search_web(state: ResearchState) -> Command:
    """
    Use LLM to generate search queries and perform web search.
    
    Equivalent to PocketFlow's SearchWeb node.
    """
    question = state["question"]
    research_context = state["research_context"]
    search_queries = state["search_queries"]
    
    print(f"🔍 Generating search query with LLM...")
    
    # Use LLM to generate better search queries
    search_prompt = f"""You are a research assistant. Generate a specific, targeted search query to help answer this question.

Question: {question}

Previous searches performed:
{chr(10).join(search_queries) if search_queries else "None"}

Current research context:
{chr(10).join(research_context) if research_context else "None"}

Generate ONE specific search query that would find the most relevant information to answer the question.
The query should be different from previous searches and fill gaps in current knowledge.

Respond with only the search query, no explanations."""

    try:
        search_query = call_llm(search_prompt).strip()
        # Clean up the search query
        search_query = search_query.replace('"', '').replace("'", "").strip()
        
    except Exception as e:
        print(f"⚠️  LLM search query generation failed: {e}")
        # Fallback to simple query generation
        search_query = generate_search_query_fallback(question, research_context, search_queries)
    
    # Simulate web search (replace with actual search API)
    print(f"🔍 Searching for: '{search_query}'")
    search_results = simulate_web_search(search_query)
    
    # Update state with search results
    updated_context = research_context + [f"Search '{search_query}': {search_results}"]
    updated_queries = search_queries + [search_query]
    
    print(f" Found: {search_results[:100]}{'...' if len(search_results) > 100 else ''}")
    
    return Command(
        update={
            "research_context": updated_context,
            "search_queries": updated_queries,
            "searches_performed": state["searches_performed"] + 1,
            "reasoning_steps": state["reasoning_steps"] + [f"Performed search: '{search_query}'"]
        },
        goto="analyze_question"  # Go back to analysis
    )

def generate_search_query_fallback(question: str, context: List[str], previous_queries: List[str]) -> str:
    """Fallback search query generation when LLM fails."""
    # Simple keyword extraction from question
    words = re.findall(r'\b\w+\b', question.lower())
    # Remove common words
    stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    if len(keywords) >= 2:
        return ' '.join(keywords[:3])  # Use top 3 keywords
    else:
        return question[:50]  # Use first part of question

def generate_answer(state: ResearchState) -> Command:
    """
    Generate final answer using LLM based on all research context.
    
    Equivalent to PocketFlow's GenerateAnswer node.
    """
    question = state["question"]
    research_context = state["research_context"]
    reasoning_steps = state["reasoning_steps"]
    
    print("💡 Generating comprehensive answer with LLM...")
    
    # Create comprehensive context for LLM
    context_text = "\n\n".join(research_context) if research_context else "No research data available."
    reasoning_text = "\n".join(f"- {step}" for step in reasoning_steps)
    
    answer_prompt = f"""You are a research assistant providing a comprehensive answer based on gathered information.

Question: {question}

Research Data:
{context_text}

Research Process:
{reasoning_text}

Based on the research data above, provide a comprehensive, well-structured answer to the question.
If the research data is insufficient, acknowledge what information is missing.
Structure your response clearly with relevant details and examples from the research.

Answer:"""

    try:
        answer = call_llm(answer_prompt)
        
        # Extract just the answer part if LLM included extra text
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
    except Exception as e:
        print(f"⚠️  LLM answer generation failed: {e}")
        # Fallback answer generation
        answer = generate_answer_fallback(question, research_context)
    
    print("📝 Research completed!")
    print("="*60)
    print(f"QUESTION: {question}")
    print("="*60)
    print(f"ANSWER: {answer}")
    print("="*60)
    
    return Command(
        update={
            "answer": answer,
            "reasoning_steps": reasoning_steps + ["Generated final answer using LLM"]
        },
        goto=END
    )

def generate_answer_fallback(question: str, research_context: List[str]) -> str:
    """Fallback answer generation when LLM fails."""
    if not research_context:
        return f"I couldn't find sufficient information to answer: {question}"
    
    # Simple text combination
    context_summary = " ".join(research_context)[:500]
    return f"Based on available research: {context_summary}..."
    """
    Generate the final answer based on accumulated research.
    
    Equivalent to PocketFlow's AnswerQuestion node.
    """
    question = state["question"]
    research_context = state["research_context"]
    confidence = state["confidence_level"]
    
    # Generate comprehensive answer
    answer = synthesize_answer(question, research_context, confidence)
    
    print(f"\n🎯 Final Answer (confidence: {confidence:.1%}):")
    print(answer)
    
    return {
        "final_answer": answer,
        "reasoning_steps": state["reasoning_steps"] + ["Generated final answer"]
    }

def calculate_confidence(question: str, research_context: List[str]) -> float:
    """Calculate confidence level based on available information."""
    if len(research_context) == 0:
        return 0.1
    
    # Simple heuristic: more research = higher confidence, up to a point
    base_confidence = min(0.3 + (len(research_context) * 0.2), 0.9)
    
    # Boost confidence for specific topics we have good info about
    question_lower = question.lower()
    context_text = " ".join(research_context).lower()
    
    if any(keyword in question_lower for keyword in ["nobel", "prize", "award"]):
        if "nobel" in context_text or "prize" in context_text:
            base_confidence += 0.2
    
    if any(keyword in question_lower for keyword in ["physics", "chemistry", "biology"]):
        if any(science in context_text for science in ["physics", "chemistry", "biology", "science"]):
            base_confidence += 0.2
    
    return min(base_confidence, 1.0)

def generate_search_query(question: str, research_context: List[str], past_queries: List[str]) -> str:
    """Generate a search query based on the question and what we've already searched."""
    # Extract key terms from question
    question_terms = extract_key_terms(question)
    
    # Start with the most important terms
    if len(past_queries) == 0:
        # First search: use the main question terms
        return " ".join(question_terms[:3])
    
    elif len(past_queries) == 1:
        # Second search: add more specific terms
        return f"{' '.join(question_terms[:2])} latest news 2024"
    
    else:
        # Subsequent searches: try different angles
        search_variations = [
            f"{question_terms[0]} details explanation",
            f"{question_terms[0]} recent developments",
            f"{question_terms[0]} expert analysis",
            f"{question_terms[0]} controversy debate"
        ]
        
        # Return a variation we haven't used
        for variation in search_variations:
            if variation not in past_queries:
                return variation
        
        # Fallback
        return f"{question_terms[0]} comprehensive overview"

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text for search queries."""
    # Simple extraction - in practice, you'd use NLP
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "was", "are", "were", "what", "who", "when", "where", "why", "how"}
    
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stop words and short words
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return key_terms[:5]  # Return top 5 terms

def simulate_web_search(query: str) -> str:
    """
    Simulate web search results. Replace with actual search API.
    """
    # Mock search results based on query
    if "nobel" in query.lower():
        return "The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton for foundational discoveries in machine learning. Their work on artificial neural networks has enabled breakthroughs in artificial intelligence."
    
    elif "physics" in query.lower():
        return "Recent developments in physics include advances in quantum computing, gravitational wave detection, and particle physics research at the LHC. The field continues to explore fundamental questions about the universe."
    
    elif "ai" in query.lower() or "artificial intelligence" in query.lower():
        return "Artificial Intelligence has seen rapid advancement in 2024, with improvements in large language models, computer vision, and robotics. Key developments include better reasoning capabilities and more efficient training methods."
    
    elif "climate" in query.lower():
        return "Climate science research shows continued warming trends, with 2024 marking another year of record temperatures. Scientists are focusing on mitigation strategies and adaptation measures."
    
    else:
        return f"Search results for '{query}' include recent articles, research papers, and expert analysis. Information suggests ongoing developments in this area with multiple perspectives from the scientific community."

def synthesize_answer(question: str, research_context: List[str], confidence: float) -> str:
    """Synthesize a comprehensive answer from research context."""
    if confidence < 0.3:
        return f"I don't have enough reliable information to answer '{question}' comprehensively. Based on limited research, I can only provide a general response that this topic requires more specific investigation."
    
    # Extract key information from research context
    key_facts = []
    for context in research_context:
        # Simple extraction - in practice, you'd use more sophisticated NLP
        if "Search" in context:
            fact = context.split(": ", 1)[-1]  # Get everything after the search query
            key_facts.append(fact)
    
    if not key_facts:
        return f"Based on my analysis of '{question}', I was unable to gather sufficient specific information to provide a detailed answer."
    
    # Build comprehensive answer
    answer_parts = [
        f"Based on my research, here's what I found about '{question}':",
        "",
        "Key findings:"
    ]
    
    for i, fact in enumerate(key_facts, 1):
        answer_parts.append(f"{i}. {fact}")
    
    answer_parts.extend([
        "",
        f"This information comes from {len(key_facts)} search{'es' if len(key_facts) > 1 else ''} with {confidence:.1%} confidence.",
        "Please verify these details from authoritative sources for critical decisions."
    ])
    
    return "\n".join(answer_parts)

def route_after_analysis(state: ResearchState) -> str:
    """
    Decide whether to search for more information or generate the answer.
    Uses the last_decision from the LLM analysis.
    """
    searches_performed = state["searches_performed"]
    max_searches = state["max_searches"]
    last_decision = state.get("last_decision", "SEARCH")
    
    # Safety check: don't exceed max searches
    if searches_performed >= max_searches:
        return "generate_answer"
    
    # Use LLM decision if available
    if last_decision == "SEARCH":
        return "search_web"
    else:
        return "generate_answer"

def build_research_agent():
    """Build the research agent graph."""
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("search_web", search_web)
    graph.add_node("generate_answer", generate_answer)
    
    # Add conditional routing
    graph.add_conditional_edges("analyze_question", route_after_analysis)
    graph.add_edge("search_web", "analyze_question")  # Return to analysis after search
    graph.add_edge("generate_answer", END)
    
    # Set entry point
    graph.set_entry_point("analyze_question")
    
    return graph.compile()

def main():
    """Main function to run the research agent."""
    print("GraphFlow Research Agent Example")
    print("=" * 40)
    
    # Build the agent
    research_agent = build_research_agent()
    
    # Test questions
    test_questions = [
        "Who won the Nobel Prize in Physics 2024?",
        "What are the latest developments in quantum computing?",
        "How is climate change affecting polar ice caps?",
        "What is the current state of artificial intelligence research?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)
        
        # Initialize state
        initial_state = {
            "question": question,
            "research_context": [],
            "search_queries": [],
            "current_analysis": "",
            "confidence_level": 0.0,
            "max_searches": 3,
            "searches_performed": 0,
            "final_answer": None,
            "reasoning_steps": [],
            "last_decision": "",
            "suggested_queries": [],
            "answer": ""
        }
        
        # Run the research agent
        try:
            result = research_agent.invoke(initial_state)
            
            print(f"\nResearch Summary:")
            print(f"- Searches performed: {result['searches_performed']}")
            print(f"- Final confidence: {result['confidence_level']:.1%}")
            print(f"- Reasoning steps: {len(result['reasoning_steps'])}")
            
        except Exception as e:
            print(f"Error processing question: {e}")

def interactive_mode():
    """Interactive mode for custom questions."""
    print("\nInteractive Research Agent")
    print("Ask any question and I'll research it for you!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    research_agent = build_research_agent()
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using the research agent!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        initial_state = {
            "question": question,
            "research_context": [],
            "search_queries": [],
            "current_analysis": "",
            "confidence_level": 0.0,
            "max_searches": 3,
            "searches_performed": 0,
            "final_answer": None,
            "reasoning_steps": [],
            "last_decision": "",
            "suggested_queries": [],
            "answer": ""
        }
        
        try:
            result = research_agent.invoke(initial_state)
            print(f"\n📊 Research completed with {result['confidence_level']:.1%} confidence")
            print(f"🔍 Performed {result['searches_performed']} searches")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

"""
GraphFlow Research Agent Benefits:

1. Intelligent Decision Making:
   - Confidence-based routing decisions
   - Adaptive search strategies
   - Comprehensive analysis workflow

2. State Management:
   - Accumulated research context
   - Search query tracking
   - Reasoning step logging

3. Extensibility:
   - Easy to plug in real search APIs
   - Configurable search limits
   - Modular answer synthesis

4. Error Handling:
   - Graceful degradation with low confidence
   - Search limit protection
   - Exception handling

5. Testing Support:
   - Demo mode with test questions
   - Interactive mode for custom queries
   - Detailed logging and reasoning

Usage:
# Demo mode
python 04-research-agent.py

# Interactive mode  
python 04-research-agent.py --interactive

To extend this example:
1. Integrate real search APIs (Google, Bing, etc.)
2. Add more sophisticated NLP for query generation
3. Implement source credibility scoring
4. Add citation tracking and reference generation
5. Support multiple search engines and result fusion
"""
