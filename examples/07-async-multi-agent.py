"""
GraphFlow Example: Async Multi-Agent Game

This example demonstrates:
- Async concurrent execution with multiple agents
- Inter-agent communication through queues
- Game-like interaction with turns and responses
- Complex async state management
- Graceful termination conditions

Implementation pattern:
- AsyncHinter and AsyncGuesser nodes
- AsyncFlow with concurrent execution
- Queue-based communication between agents
- Game loop with termination conditions

GraphFlow adaptation:
- Async node functions with proper state management
- Command-based flow control in async context
- Better error handling and game state tracking
- Type-safe async communication patterns
"""

import sys
import os
import asyncio

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, call_llm, configure_llm
from typing import TypedDict, List, Dict, Optional, Any
import random
import time

# State schema for async multi-agent game
class GameState(TypedDict):
    # Game setup
    target_word: str
    forbidden_words: List[str]
    max_rounds: int
    current_round: int

    # Game state
    game_status: str  # "playing", "won", "lost", "ended"
    winner: Optional[str]
    game_over: bool

    # Communication
    hinter_queue: Any  # asyncio.Queue
    guesser_queue: Any  # asyncio.Queue

    # Game history
    hints_given: List[str]
    guesses_made: List[str]
    past_wrong_guesses: List[str]

    # Agent states
    hinter_active: bool
    guesser_active: bool

    # Timing and stats
    start_time: float
    end_time: Optional[float]
    total_turns: int

async def setup_llm_for_game(state: GameState) -> Command:
    """Configure LLM provider for the multi-agent game."""
    print("🔧 Setting up LLM provider for game agents...")

    if os.environ.get("OPENAI_API_KEY"):
        configure_llm("openai", model="gpt-4")
        print("✅ Using OpenAI GPT-4 for game agents")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        configure_llm("anthropic", model="claude-3-sonnet-20240229")
        print("✅ Using Anthropic Claude for game agents")
    else:
        try:
            configure_llm("ollama", model="llama2")
            call_llm("test")
            print("✅ Using Ollama (local) for game agents")
        except Exception:
            print("❌ No LLM provider available - using fallback strategies")
            print("\nTo use AI-powered agents, you need:")
            print("1. Set OPENAI_API_KEY environment variable, OR")
            print("2. Set ANTHROPIC_API_KEY environment variable, OR")
            print("3. Start Ollama server: ollama serve")

    return Command(update={})

async def initialize_game(state: GameState) -> Command:
    """Initialize the game with queues and starting conditions."""
    print("🎮 Initializing Taboo Game...")
    print(f"Target word: {state['target_word']}")
    print(f"Forbidden words: {', '.join(state['forbidden_words'])}")
    print("=" * 50)

    # Create communication queues
    hinter_queue = asyncio.Queue()
    guesser_queue = asyncio.Queue()

    # Send initial empty message to start the game
    await hinter_queue.put("")

    return Command(
        update={
            "hinter_queue": hinter_queue,
            "guesser_queue": guesser_queue,
            "game_status": "playing",
            "hinter_active": True,
            "guesser_active": True,
            "start_time": time.time(),
            "hints_given": [],
            "guesses_made": [],
            "past_wrong_guesses": []
        }
    )

async def hinter_agent(state: GameState) -> Command:
    """
    Hinter agent that provides hints for the target word.

    """
    if not state["hinter_active"] or state["game_over"]:
        return Command(update={"hinter_active": False}, goto=END)

    try:
        # Wait for message from guesser (or empty string at start)
        guess = await asyncio.wait_for(state["hinter_queue"].get(), timeout=30.0)

        if guess == "GAME_OVER":
            return Command(
                update={"hinter_active": False, "game_status": "ended"},
                goto=END
            )

        # Generate hint based on current game state
        hint = await generate_hint(
            state["target_word"],
            state["forbidden_words"],
            state["past_wrong_guesses"],
            state["hints_given"]
        )

        print(f"\n🎯 Hinter: {hint}")

        # Send hint to guesser
        await state["guesser_queue"].put(hint)

        return Command(
            update={
                "hints_given": state["hints_given"] + [hint],
                "current_round": state["current_round"] + 1,
                "total_turns": state["total_turns"] + 1
            },
            goto="check_game_state"
        )

    except asyncio.TimeoutError:
        print("⏰ Hinter timeout - ending game")
        return Command(
            update={
                "game_status": "timeout",
                "game_over": True,
                "hinter_active": False
            },
            goto=END
        )
    except Exception as e:
        print(f"❌ Hinter error: {e}")
        return Command(
            update={
                "game_status": "error", 
                "game_over": True,
                "hinter_active": False
            },
            goto=END
        )

async def guesser_agent(state: GameState) -> Command:
    """
    Guesser agent that makes guesses based on hints.

    """
    if not state["guesser_active"] or state["game_over"]:
        return Command(update={"guesser_active": False}, goto=END)

    try:
        # Wait for hint from hinter
        hint = await asyncio.wait_for(state["guesser_queue"].get(), timeout=30.0)

        # Generate guess based on hint and game history
        guess = await generate_guess(
            hint,
            state["past_wrong_guesses"],
            state["guesses_made"]
        )

        print(f"🤔 Guesser: I think it's '{guess}'")

        # Check if guess is correct
        is_correct = guess.lower().strip() == state["target_word"].lower().strip()

        if is_correct:
            print("🎉 Correct! Game won!")

            # Notify hinter that game is over
            await state["hinter_queue"].put("GAME_OVER")

            return Command(
                update={
                    "game_status": "won",
                    "winner": "guesser",
                    "game_over": True,
                    "guesser_active": False,
                    "guesses_made": state["guesses_made"] + [guess],
                    "end_time": time.time()
                },
                goto=END
            )
        else:
            print(f"❌ Incorrect guess: '{guess}'")

            # Check if max rounds reached
            if state["current_round"] >= state["max_rounds"]:
                print("⏱️ Max rounds reached - game over!")
                await state["hinter_queue"].put("GAME_OVER")

                return Command(
                    update={
                        "game_status": "lost",
                        "winner": "hinter",
                        "game_over": True,
                        "guesser_active": False,
                        "guesses_made": state["guesses_made"] + [guess],
                        "past_wrong_guesses": state["past_wrong_guesses"] + [guess],
                        "end_time": time.time()
                    },
                    goto=END
                )

            # Continue game - send guess to hinter
            await state["hinter_queue"].put(guess)

            return Command(
                update={
                    "guesses_made": state["guesses_made"] + [guess],
                    "past_wrong_guesses": state["past_wrong_guesses"] + [guess],
                    "total_turns": state["total_turns"] + 1
                },
                goto="check_game_state"
            )

    except asyncio.TimeoutError:
        print("⏰ Guesser timeout - ending game")
        return Command(
            update={
                "game_status": "timeout",
                "game_over": True,
                "guesser_active": False
            },
            goto=END
        )
    except Exception as e:
        print(f"❌ Guesser error: {e}")
        return Command(
            update={
                "game_status": "error",
                "game_over": True,
                "guesser_active": False
            },
            goto=END
        )

async def check_game_state(state: GameState) -> Command:
    """Check game state and decide whether to continue."""
    if state["game_over"]:
        return Command(update={}, goto="end_game")

    if state["current_round"] >= state["max_rounds"]:
        return Command(
            update={
                "game_status": "max_rounds",
                "game_over": True,
                "winner": "hinter"
            },
            goto="end_game"
        )

    # Game continues - both agents should continue their loops
    return Command(update={}, goto="continue_game")

async def continue_game(state: GameState) -> Command:
    """Signal that game should continue."""
    return Command(update={"game_continuing": True})

async def end_game(state: GameState) -> Command:
    """Handle game ending and show results."""
    end_time = state.get("end_time", time.time())
    duration = end_time - state["start_time"]

    print(f"\n🏁 Game Over!")
    print(f"Status: {state['game_status']}")
    print(f"Winner: {state.get('winner', 'None')}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Total turns: {state['total_turns']}")
    print(f"Hints given: {len(state['hints_given'])}")
    print(f"Guesses made: {len(state['guesses_made'])}")

    if state["hints_given"]:
        print(f"\nHints: {', '.join(state['hints_given'])}")
    if state["guesses_made"]:
        print(f"Guesses: {', '.join(state['guesses_made'])}")

    return Command(update={"final_stats_displayed": True})

async def generate_hint(target_word: str, forbidden_words: List[str], 
                       past_guesses: List[str], previous_hints: List[str]) -> str:
    """Generate a hint for the target word avoiding forbidden words using LLM."""

    # Simulate AI thinking time
    await asyncio.sleep(0.5)

    # Create context for LLM
    forbidden_text = ', '.join(forbidden_words)
    wrong_guesses_text = ', '.join(past_guesses) if past_guesses else "None"
    previous_hints_text = '\n'.join(f"- {hint}" for hint in previous_hints) if previous_hints else "None"

    hint_prompt = f"""You are playing Taboo! You need to give a hint to help someone guess the target word.

TARGET WORD: {target_word}

RULES:
- You CANNOT use any of these forbidden words in your hint: {forbidden_text}
- You cannot use the target word itself
- Give only ONE short, creative hint (max 10 words)
- Be clever and think of synonyms or descriptions

GAME CONTEXT:
Wrong guesses so far: {wrong_guesses_text}
Previous hints you gave:
{previous_hints_text}

Give a helpful hint that doesn't repeat your previous hints and avoids all forbidden words.

Hint:"""

    try:
        # Try to use LLM for intelligent hint generation
        hint = call_llm(hint_prompt).strip()

        # Clean up the response
        if hint.startswith("Hint:"):
            hint = hint[5:].strip()
        if hint.startswith('"') and hint.endswith('"'):
            hint = hint[1:-1]

        # Validate hint doesn't contain forbidden words
        hint_words = hint.lower().split()
        forbidden_found = any(forbidden.lower() in ' '.join(hint_words) for forbidden in forbidden_words)

        if forbidden_found:
            print(f"⚠️  LLM hint contained forbidden words, using fallback")
            return generate_hint_fallback(target_word, forbidden_words, past_guesses, previous_hints)

        print(f"🤖 LLM generated hint: {hint}")
        return hint

    except Exception as e:
        print(f"⚠️  LLM hint generation failed: {e}")
        return generate_hint_fallback(target_word, forbidden_words, past_guesses, previous_hints)

def generate_hint_fallback(target_word: str, forbidden_words: List[str], 
                         past_guesses: List[str], previous_hints: List[str]) -> str:
    """Fallback hint generation when LLM is unavailable."""

    # Simple hint generation logic
    hints_by_word = {
        "nostalgic": [
            "Feeling from remembering good times",
            "Emotion when looking at old photos", 
            "Wistful about the past",
            "Sentimental about bygone days",
            "Yearning for earlier times"
        ],
        "adventure": [
            "Exciting journey or experience",
            "Bold expedition or quest",
            "Thrilling exploration", 
            "Daring escapade",
            "Epic journey"
        ],
        "mysterious": [
            "Hard to understand or explain",
            "Enigmatic and puzzling",
            "Full of secrets",
            "Unexplained phenomenon", 
            "Cryptic and unknown"
        ],
        "serendipity": [
            "Happy accident or coincidence",
            "Fortunate discovery by chance",
            "Lucky find when looking for something else",
            "Pleasant surprise encounter",
            "Unexpected good fortune"
        ]
    }

    available_hints = hints_by_word.get(target_word.lower(), [
        "Think of a common word",
        "It's something you might encounter daily",
        "Consider different meanings",
        "What comes to mind first?",
        "It's more specific than you think"
    ])

    # Filter out hints that use forbidden words
    filtered_hints = []
    for hint in available_hints:
        hint_words = hint.lower().split()
        if not any(forbidden.lower() in hint_words for forbidden in forbidden_words):
            filtered_hints.append(hint)

    # Avoid repeating previous hints
    new_hints = [h for h in filtered_hints if h not in previous_hints]

    if new_hints:
        return random.choice(new_hints)
    elif filtered_hints:
        return random.choice(filtered_hints) + " (repeated)"
    else:
        return "It's a word you know (can't give better hint without forbidden words)"

async def generate_guess(hint: str, past_wrong_guesses: List[str], 
                        all_guesses: List[str]) -> str:
    """Generate a guess based on the hint and game history using LLM."""

    # Simulate AI thinking time
    await asyncio.sleep(0.3)

    # Create context for LLM
    wrong_guesses_text = ', '.join(past_wrong_guesses) if past_wrong_guesses else "None"

    guess_prompt = f"""You are playing Taboo! You need to guess a word based on the hint given.

HINT: "{hint}"

CONTEXT:
- This is a word guessing game
- Wrong guesses so far: {wrong_guesses_text}
- Don't repeat any wrong guesses

Based on the hint, what word do you think it is? Give only ONE word as your guess.

Guess:"""

    try:
        # Try to use LLM for intelligent guess generation
        guess = call_llm(guess_prompt).strip()

        # Clean up the response
        if guess.startswith("Guess:"):
            guess = guess[6:].strip()
        if guess.startswith('"') and guess.endswith('"'):
            guess = guess[1:-1]

        # Take only the first word if multiple words returned
        guess = guess.split()[0].lower() if guess.split() else guess.lower()

        # Make sure it's not a repeated wrong guess
        if guess not in past_wrong_guesses:
            print(f"🤖 LLM guessed: {guess}")
            return guess
        else:
            print(f"⚠️  LLM repeated a wrong guess, using fallback")
            return generate_guess_fallback(hint, past_wrong_guesses, all_guesses)

    except Exception as e:
        print(f"⚠️  LLM guess generation failed: {e}")
        return generate_guess_fallback(hint, past_wrong_guesses, all_guesses)

def generate_guess_fallback(hint: str, past_wrong_guesses: List[str], 
                          all_guesses: List[str]) -> str:
    """Fallback guess generation when LLM is unavailable."""

    hint_lower = hint.lower()

    # Pattern matching for different hints
    if "feeling" in hint_lower or "emotion" in hint_lower:
        candidates = ["nostalgic", "happy", "sad", "excited", "peaceful"]
    elif "journey" in hint_lower or "expedition" in hint_lower:
        candidates = ["adventure", "travel", "quest", "exploration", "trip"]
    elif "mysterious" in hint_lower or "enigmatic" in hint_lower:
        candidates = ["mysterious", "secret", "hidden", "unknown", "cryptic"]
    elif "accident" in hint_lower or "coincidence" in hint_lower:
        candidates = ["serendipity", "chance", "luck", "fortune", "fate"]
    elif "photo" in hint_lower or "past" in hint_lower:
        candidates = ["nostalgic", "memory", "remembrance", "reflection"]
    else:
        # Generic guesses
        candidates = ["adventure", "mysterious", "nostalgic", "serendipity", "beautiful", "wonderful", "amazing"]

    # Filter out past wrong guesses
    candidates = [word for word in candidates if word not in past_wrong_guesses]

    if candidates:
        return random.choice(candidates)
    else:
        # Generate a random guess if all candidates exhausted
        backup_words = ["love", "life", "hope", "dream", "future", "success", "happiness", "freedom"]
        backup_words = [word for word in backup_words if word not in past_wrong_guesses]
        return random.choice(backup_words) if backup_words else "unknown"

def build_async_game_graph():
    """Build the async multi-agent game graph."""
    graph = StateGraph(GameState)

    # Add async nodes
    graph.add_node("setup_llm", setup_llm_for_game)
    graph.add_node("initialize", initialize_game)
    graph.add_node("hinter", hinter_agent)
    graph.add_node("guesser", guesser_agent)
    graph.add_node("check_state", check_game_state)
    graph.add_node("continue_game", continue_game)
    graph.add_node("end_game", end_game)

    # Set up flow
    graph.add_edge("setup_llm", "initialize")
    graph.add_edge("initialize", "check_state")
    graph.add_edge("continue_game", "check_state")
    graph.add_edge("end_game", END)

    graph.set_entry_point("setup_llm")

    return graph.compile()

async def run_concurrent_agents(game_state: GameState):
    """Run hinter and guesser agents concurrently."""

    async def run_hinter():
        while game_state["hinter_active"] and not game_state["game_over"]:
            try:
                result = await hinter_agent(game_state)
                # Update state from command
                if hasattr(result, 'update'):
                    for key, value in result.update.items():
                        game_state[key] = value
                if result.goto == END:
                    break
            except Exception as e:
                print(f"Hinter error: {e}")
                game_state["hinter_active"] = False
            await asyncio.sleep(0.1)

    async def run_guesser():
        while game_state["guesser_active"] and not game_state["game_over"]:
            try:
                result = await guesser_agent(game_state)
                # Update state from command
                if hasattr(result, 'update'):
                    for key, value in result.update.items():
                        game_state[key] = value
                if result.goto == END:
                    break
            except Exception as e:
                print(f"Guesser error: {e}")
                game_state["guesser_active"] = False
            await asyncio.sleep(0.1)

    # Run both agents concurrently
    await asyncio.gather(run_hinter(), run_guesser())

async def main():
    """Main async function to run the multi-agent game."""
    print("GraphFlow Async Multi-Agent Taboo Game")
    print("=" * 50)

    # Game configurations
    games = [
        {
            "target_word": "nostalgic",
            "forbidden_words": ["memory", "past", "remember", "feeling", "longing"],
            "max_rounds": 8
        },
        {
            "target_word": "adventure", 
            "forbidden_words": ["journey", "travel", "exciting", "exploration"],
            "max_rounds": 6
        },
        {
            "target_word": "serendipity",
            "forbidden_words": ["luck", "chance", "accident", "fortune"],
            "max_rounds": 10
        }
    ]

    for i, game_config in enumerate(games, 1):
        print(f"\n🎮 Starting Game {i}/3")
        print(f"Target: {game_config['target_word']}")
        print(f"Forbidden: {', '.join(game_config['forbidden_words'])}")
        print("-" * 50)

        # Initialize game state
        game_state = {
            "target_word": game_config["target_word"],
            "forbidden_words": game_config["forbidden_words"],
            "max_rounds": game_config["max_rounds"],
            "current_round": 0,
            "game_status": "initializing",
            "winner": None,
            "game_over": False,
            "hinter_queue": None,
            "guesser_queue": None,
            "hints_given": [],
            "guesses_made": [],
            "past_wrong_guesses": [],
            "hinter_active": False,
            "guesser_active": False,
            "start_time": 0.0,
            "end_time": None,
            "total_turns": 0
        }

        try:
            # Setup LLM for the game agents
            await setup_llm_for_game(game_state)

            # Initialize the game
            init_result = await initialize_game(game_state)
            if hasattr(init_result, 'update') and init_result.update:
                game_state.update(init_result.update)

            # Run concurrent agents (this is the main demo of async multi-agent)
            await run_concurrent_agents(game_state)

            # End game
            end_result = await end_game(game_state)

            print(f"✅ Game {i} completed!")
            if game_state.get("winner"):
                print(f"🏆 Winner: {game_state['winner']}")

        except Exception as e:
            print(f"❌ Game {i} error: {e}")

        if i < len(games):
            print("\n⏳ Next game starting in 2 seconds...")
            await asyncio.sleep(2)

    print(f"\n🏁 All games completed!")

async def interactive_game():
    """Interactive mode where user can set up custom games."""
    print("Interactive Taboo Game Setup")
    print("-" * 30)

    target_word = input("Enter target word: ").strip()
    if not target_word:
        target_word = "mystery"

    forbidden_input = input("Enter forbidden words (comma-separated): ").strip()
    forbidden_words = [word.strip() for word in forbidden_input.split(",") if word.strip()]

    try:
        max_rounds = int(input("Max rounds (default 8): ").strip() or "8")
    except ValueError:
        max_rounds = 8

    game_state = {
        "target_word": target_word,
        "forbidden_words": forbidden_words,
        "max_rounds": max_rounds,
        "current_round": 0,
        "game_status": "initializing",
        "winner": None,
        "game_over": False,
        "hinter_queue": None,
        "guesser_queue": None,
        "hints_given": [],
        "guesses_made": [],
        "past_wrong_guesses": [],
        "hinter_active": False,
        "guesser_active": False,
        "start_time": 0.0,
        "end_time": None,
        "total_turns": 0
    }

    print(f"\n🎯 Starting custom game:")
    print(f"Target: {target_word}")
    print(f"Forbidden: {', '.join(forbidden_words)}")
    print(f"Max rounds: {max_rounds}")
    print("-" * 40)

    try:
        init_updates = await initialize_game(game_state)
        game_state.update(init_updates)
        await run_concurrent_agents(game_state)
        await end_game(game_state)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_game())
    else:
        asyncio.run(main())

"""
GraphFlow Async Multi-Agent Benefits:

1. True Concurrency:
   - Agents run simultaneously using asyncio
   - Non-blocking communication via queues
   - Proper async/await patterns

2. Robust Communication:
   - Type-safe message passing
   - Timeout handling for reliability
   - Graceful error recovery

3. Game State Management:
   - Comprehensive state tracking
   - Turn-based coordination
   - Win/lose condition handling

4. Extensibility:
   - Easy to add more agent types
   - Configurable game parameters
   - Pluggable AI logic

5. Production Ready:
   - Proper error handling
   - Resource cleanup
   - Performance monitoring

Usage:
# Demo mode with predefined games
python 11-async-multi-agent.py

# Interactive mode
python 11-async-multi-agent.py --interactive

To extend this example:
1. Add real LLM integration for smarter agents
2. Implement web interface for human players
3. Add tournament mode with multiple games
4. Create learning agents that improve over time
5. Add voice interaction capabilities
"""