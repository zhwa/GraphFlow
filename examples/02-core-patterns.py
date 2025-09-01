"""
GraphFlow Core Patterns

This module demonstrates essential GraphFlow patterns with parallel execution:
- Simple workflows with state reducers
- Parallel processing patterns
- Fan-out and fan-in operations
- Conditional routing with parallelism
- Error handling in parallel contexts
- Performance optimization techniques

Each pattern showcases GraphFlow capabilities over traditional linear execution.
"""

import sys
import os
import asyncio
import time
from typing import Dict, Any, List

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, with_reducers

# Pattern 1: Simple Parallel Processing
async def pattern_simple_parallel():
    """Basic parallel processing with multiple workers."""
    print("🔄 Pattern 1: Simple Parallel Processing")
    print("-" * 50)
    
    def create_work_state(tasks: List[str]):
        return {
            "tasks": tasks,
            "results": [],
            "metadata": {},
            "completed": False
        }
    
    graph = StateGraph(
        state_reducers=with_reducers(
            results='extend',
            metadata='merge'
        )
    )
    
    def distribute_work(state: Dict[str, Any]) -> Command:
        """Distribute work to parallel workers."""
        print(f"📋 Distributing {len(state['tasks'])} tasks to workers...")
        return Command(
            update={"start_time": time.time()},
            goto=["worker_fast", "worker_medium", "worker_slow"]
        )
    
    async def worker_fast(state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast worker processing subset of tasks."""
        await asyncio.sleep(0.1)
        tasks_subset = state["tasks"][:2]
        results = [f"FAST-{task}" for task in tasks_subset]
        return {
            "results": results,
            "metadata": {"fast_worker": {"processed": len(results), "speed": "fast"}}
        }
    
    async def worker_medium(state: Dict[str, Any]) -> Dict[str, Any]:
        """Medium speed worker."""
        await asyncio.sleep(0.2)
        tasks_subset = state["tasks"][2:4]
        results = [f"MEDIUM-{task}" for task in tasks_subset]
        return {
            "results": results,
            "metadata": {"medium_worker": {"processed": len(results), "speed": "medium"}}
        }
    
    async def worker_slow(state: Dict[str, Any]) -> Dict[str, Any]:
        """Slow but thorough worker."""
        await asyncio.sleep(0.3)
        tasks_subset = state["tasks"][4:]
        results = [f"SLOW-{task}" for task in tasks_subset]
        return {
            "results": results,
            "metadata": {"slow_worker": {"processed": len(results), "speed": "slow"}}
        }
    
    def collect_results(state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and summarize results from all workers."""
        processing_time = time.time() - state["start_time"]
        return {
            "completed": True,
            "summary": {
                "total_results": len(state["results"]),
                "workers_used": len(state["metadata"]),
                "processing_time": processing_time
            }
        }
    
    # Build graph
    graph.add_node("distribute", distribute_work)
    graph.add_node("worker_fast", worker_fast)
    graph.add_node("worker_medium", worker_medium)
    graph.add_node("worker_slow", worker_slow)
    graph.add_node("collect", collect_results)
    
    # Fan-in pattern
    graph.add_edge("worker_fast", "collect")
    graph.add_edge("worker_medium", "collect")
    graph.add_edge("worker_slow", "collect")
    graph.set_entry_point("distribute")
    
    # Execute
    compiled = graph.compile()
    tasks = ["task1", "task2", "task3", "task4", "task5", "task6"]
    result = await compiled.ainvoke(create_work_state(tasks))
    
    print(f"✅ Processed {result['summary']['total_results']} results")
    print(f"⏱️  Time: {result['summary']['processing_time']:.3f}s")
    print(f"👥 Workers: {result['summary']['workers_used']}")
    
    return result

# Pattern 2: Conditional Parallel Routing
async def pattern_conditional_parallel():
    """Conditional routing with parallel execution branches.""" 
    print("\n🔀 Pattern 2: Conditional Parallel Routing")
    print("-" * 50)
    
    def create_decision_state(data_type: str, data: Any):
        return {
            "data_type": data_type,
            "data": data,
            "processing_results": [],
            "decision_path": [],
            "metadata": {}
        }
    
    graph = StateGraph(
        state_reducers=with_reducers(
            processing_results='extend',
            decision_path='extend',
            metadata='merge'
        )
    )
    
    def analyze_and_route(state: Dict[str, Any]) -> Command:
        """Analyze data and route to appropriate parallel processors."""
        data_type = state["data_type"]
        
        if data_type == "image":
            print(f"🖼️  Routing to image processing pipeline...")
            return Command(
                update={"decision_path": [f"routed_to_image_pipeline"]},
                goto=["image_preprocessor", "image_analyzer", "image_enhancer"]
            )
        elif data_type == "text":
            print(f"📝 Routing to text processing pipeline...")
            return Command(
                update={"decision_path": [f"routed_to_text_pipeline"]},
                goto=["text_tokenizer", "text_analyzer", "text_summarizer"]
            )
        else:
            print(f"🔧 Routing to general processing...")
            return Command(
                update={"decision_path": [f"routed_to_general"]},
                goto="general_processor"
            )
    
    # Image processing workers
    async def image_preprocessor(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "processing_results": [{"worker": "image_preprocessor", "result": "preprocessed_image"}],
            "metadata": {"preprocessing_time": time.time()}
        }
    
    async def image_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        return {
            "processing_results": [{"worker": "image_analyzer", "result": "image_features_extracted"}],
            "metadata": {"analysis_time": time.time()}
        }
    
    async def image_enhancer(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.15)
        return {
            "processing_results": [{"worker": "image_enhancer", "result": "enhanced_image"}],
            "metadata": {"enhancement_time": time.time()}
        }
    
    # Text processing workers
    async def text_tokenizer(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        text = state["data"]
        tokens = len(text.split()) if isinstance(text, str) else 0
        return {
            "processing_results": [{"worker": "text_tokenizer", "result": f"tokenized_{tokens}_words"}],
            "metadata": {"tokenization_time": time.time()}
        }
    
    async def text_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "processing_results": [{"worker": "text_analyzer", "result": "text_analysis_complete"}],
            "metadata": {"text_analysis_time": time.time()}
        }
    
    async def text_summarizer(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.12)
        return {
            "processing_results": [{"worker": "text_summarizer", "result": "text_summary_generated"}],
            "metadata": {"summarization_time": time.time()}
        }
    
    def general_processor(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "processing_results": [{"worker": "general", "result": f"processed_{state['data_type']}"}],
            "metadata": {"general_processing_time": time.time()}
        }
    
    def finalize_processing(state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize processing results from parallel workers."""
        return {
            "decision_path": ["finalization_complete"],
            "final_summary": {
                "workers_executed": len(state["processing_results"]),
                "data_type_processed": state["data_type"],
                "total_steps": len(state["decision_path"])
            }
        }
    
    # Build graph
    graph.add_node("analyze", analyze_and_route)
    
    # Image pipeline
    graph.add_node("image_preprocessor", image_preprocessor)
    graph.add_node("image_analyzer", image_analyzer)
    graph.add_node("image_enhancer", image_enhancer)
    
    # Text pipeline
    graph.add_node("text_tokenizer", text_tokenizer)
    graph.add_node("text_analyzer", text_analyzer)
    graph.add_node("text_summarizer", text_summarizer)
    
    # General processor
    graph.add_node("general_processor", general_processor)
    
    # Finalization
    graph.add_node("finalize", finalize_processing)
    
    # Fan-in edges
    graph.add_edge("image_preprocessor", "finalize")
    graph.add_edge("image_analyzer", "finalize")
    graph.add_edge("image_enhancer", "finalize")
    graph.add_edge("text_tokenizer", "finalize")
    graph.add_edge("text_analyzer", "finalize")
    graph.add_edge("text_summarizer", "finalize")
    graph.add_edge("general_processor", "finalize")
    graph.set_entry_point("analyze")
    
    compiled = graph.compile()
    
    # Test different data types
    test_cases = [
        ("image", "sample_image.jpg"),
        ("text", "This is a sample text for processing with multiple words and sentences."),
        ("data", {"numbers": [1, 2, 3], "metadata": "test"})
    ]
    
    for data_type, data in test_cases:
        print(f"🧪 Testing {data_type} processing...")
        result = await compiled.ainvoke(create_decision_state(data_type, data))
        print(f"   📊 Workers: {result['final_summary']['workers_executed']}")
        print(f"   🛤️  Path: {' -> '.join(result['decision_path'])}")
    
    return True

# Pattern 3: Pipeline with State Transformation
async def pattern_pipeline_transformation():
    """Data pipeline with parallel transformation stages."""
    print("\n🔄 Pattern 3: Pipeline with State Transformation")
    print("-" * 50)
    
    def create_pipeline_state(raw_data: List[Dict]):
        return {
            "raw_data": raw_data,
            "validated_data": [],
            "enriched_data": [],
            "processed_data": [],
            "pipeline_metadata": {}
        }
    
    graph = StateGraph(
        state_reducers=with_reducers(
            validated_data='extend',
            enriched_data='extend', 
            processed_data='extend',
            pipeline_metadata='merge'
        )
    )
    
    def start_pipeline(state: Dict[str, Any]) -> Command:
        """Start the data processing pipeline."""
        print(f"🚀 Starting pipeline with {len(state['raw_data'])} records...")
        return Command(
            update={"pipeline_metadata": {"start_time": time.time()}},
            goto=["validator", "enricher", "processor"]  # Parallel processing stages
        )
    
    async def validator(state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data records in parallel."""
        await asyncio.sleep(0.1)
        valid_records = []
        
        for record in state["raw_data"]:
            if "id" in record and "value" in record:
                valid_records.append({**record, "validated": True})
        
        return {
            "validated_data": valid_records,
            "pipeline_metadata": {
                "validation_time": time.time(),
                "validation_count": len(valid_records)
            }
        }
    
    async def enricher(state: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with additional information."""
        await asyncio.sleep(0.15)
        enriched_records = []
        
        for record in state["raw_data"]:
            enriched_record = {
                **record,
                "enriched": True,
                "enrichment_timestamp": time.time(),
                "enriched_value": record.get("value", 0) * 2
            }
            enriched_records.append(enriched_record)
        
        return {
            "enriched_data": enriched_records,
            "pipeline_metadata": {
                "enrichment_time": time.time(),
                "enrichment_count": len(enriched_records)
            }
        }
    
    async def processor(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with business logic."""
        await asyncio.sleep(0.12)
        processed_records = []
        
        for record in state["raw_data"]:
            processed_record = {
                "id": record.get("id"),
                "processed_value": record.get("value", 0) ** 2,
                "processed": True,
                "processing_timestamp": time.time()
            }
            processed_records.append(processed_record)
        
        return {
            "processed_data": processed_records,
            "pipeline_metadata": {
                "processing_time": time.time(),
                "processing_count": len(processed_records)
            }
        }
    
    def merge_pipeline_results(state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from all pipeline stages."""
        total_time = time.time() - state["pipeline_metadata"]["start_time"]
        
        return {
            "pipeline_complete": True,
            "final_summary": {
                "input_records": len(state["raw_data"]),
                "validated_records": len(state["validated_data"]),
                "enriched_records": len(state["enriched_data"]),
                "processed_records": len(state["processed_data"]),
                "total_pipeline_time": total_time,
                "stages_completed": 3
            }
        }
    
    # Build graph
    graph.add_node("start", start_pipeline)
    graph.add_node("validator", validator)
    graph.add_node("enricher", enricher)
    graph.add_node("processor", processor)
    graph.add_node("merge", merge_pipeline_results)
    
    # Fan-in pattern
    graph.add_edge("validator", "merge")
    graph.add_edge("enricher", "merge")
    graph.add_edge("processor", "merge")
    graph.set_entry_point("start")
    
    compiled = graph.compile()
    
    # Test data
    test_data = [
        {"id": 1, "value": 10, "type": "A"},
        {"id": 2, "value": 20, "type": "B"},
        {"id": 3, "value": 30, "type": "A"},
        {"id": 4, "value": 40, "type": "C"}
    ]
    
    result = await compiled.ainvoke(create_pipeline_state(test_data))
    
    print(f"✅ Pipeline completed!")
    print(f"📊 Summary: {result['final_summary']}")
    print(f"⏱️  Total time: {result['final_summary']['total_pipeline_time']:.3f}s")
    
    return result

async def run_core_patterns():
    """Run all core GraphFlow patterns."""
    print("GraphFlow Core Patterns Demonstration")
    print("=" * 60)
    
    patterns = [
        ("Simple Parallel Processing", pattern_simple_parallel),
        ("Conditional Parallel Routing", pattern_conditional_parallel),
        ("Pipeline with State Transformation", pattern_pipeline_transformation)
    ]
    
    results = []
    for name, pattern_func in patterns:
        try:
            print(f"\n🎯 Running: {name}")
            result = await pattern_func()
            results.append(True)
            print(f"✅ {name} completed successfully!")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Core Patterns Summary:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total-passed}/{total}")
    
    if passed == total:
        print(f"\n🎉 All core patterns executed successfully!")
        print("🚀 GraphFlow parallel execution is working perfectly!")
    else:
        print(f"\n⚠️  Some patterns failed. Check the output above.")
    
    print(f"\n🎯 Key Patterns Demonstrated:")
    print("   • Fan-out: One node triggers multiple parallel workers")
    print("   • Fan-in: Multiple workers converge to single result")
    print("   • State reducers: Automatic concurrent state merging")
    print("   • Conditional routing: Dynamic parallel execution paths")
    print("   • Pipeline processing: Multi-stage parallel data transformation")

if __name__ == "__main__":
    asyncio.run(run_core_patterns())
