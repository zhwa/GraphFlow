#!/usr/bin/env python3
"""
Map-Reduce Pattern with GraphFlow
=================================

This example demonstrates the classic Map-Reduce pattern using GraphFlow's
parallel execution engine. Map-Reduce is perfect for processing large datasets
by distributing work across multiple workers (Map phase) and then combining
results (Reduce phase).

Key Concepts Demonstrated:
- Fan-out to multiple parallel mappers
- Independent data processing
- Fan-in to a single reducer
- State management with 'extend' reducer
- Performance comparison vs sequential processing
"""

import sys
import os
import time

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command

def create_map_reduce_processor():
    """
    Creates a Map-Reduce workflow that processes a list of numbers:
    1. Split data into chunks (Distribute)
    2. Process each chunk in parallel (Map phase)
    3. Combine all results (Reduce phase)
    """

    # Configure state management for parallel execution
    graph = StateGraph(state_reducers={
        'mapped_results': 'extend',  # Collect results from all mappers
        'processing_log': 'extend',  # Track processing steps
        'chunk_info': 'extend'       # Store chunk metadata
    })

    def distribute_data(state):
        """Split input data into chunks for parallel processing"""
        data = state.get('input_data', [])
        chunk_size = state.get('chunk_size', 3)

        # Split data into chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)

        print(f"📊 Distributing {len(data)} items into {len(chunks)} chunks of size ~{chunk_size}")

        # Fan-out to multiple mappers based on number of chunks
        mapper_nodes = [f'mapper_{i}' for i in range(len(chunks))]

        return Command(
            update={
                'chunks': chunks,
                'num_chunks': len(chunks),
                'processing_log': [f'Split {len(data)} items into {len(chunks)} chunks'],
                'start_time': time.time()
            },
            goto=mapper_nodes  # Dynamic fan-out based on data size
        )

    def create_mapper(mapper_id):
        """Factory function to create mapper workers"""
        def mapper_function(state):
            """Map function: Process assigned chunk of data"""
            chunks = state.get('chunks', [])

            if mapper_id >= len(chunks):
                # No chunk assigned to this mapper
                return {
                    'processing_log': [f'Mapper {mapper_id}: No chunk assigned'],
                    'chunk_info': [{'mapper_id': mapper_id, 'status': 'no_data'}]
                }

            chunk = chunks[mapper_id]
            print(f"🔄 Mapper {mapper_id} processing chunk: {chunk}")

            # Simulate processing time
            time.sleep(0.2)

            # Example processing: square each number and sum the chunk
            processed_chunk = [x ** 2 for x in chunk]
            chunk_sum = sum(processed_chunk)

            return {
                'mapped_results': [{
                    'mapper_id': mapper_id,
                    'original_chunk': chunk,
                    'processed_chunk': processed_chunk,
                    'chunk_sum': chunk_sum,
                    'chunk_size': len(chunk)
                }],
                'processing_log': [f'Mapper {mapper_id}: Processed {len(chunk)} items → sum={chunk_sum}'],
                'chunk_info': [{
                    'mapper_id': mapper_id, 
                    'chunk_size': len(chunk),
                    'chunk_sum': chunk_sum,
                    'status': 'completed'
                }]
            }

        return mapper_function

    def reduce_results(state):
        """Reduce function: Combine all mapper results"""
        mapped_results = state.get('mapped_results', [])
        start_time = state.get('start_time', time.time())
        processing_time = time.time() - start_time

        print(f"🔗 Reducer combining results from {len(mapped_results)} mappers...")

        # Combine all results
        total_sum = 0
        total_items = 0
        all_processed_values = []

        for result in mapped_results:
            total_sum += result['chunk_sum']
            total_items += result['chunk_size']
            all_processed_values.extend(result['processed_chunk'])

        # Calculate statistics
        avg_value = total_sum / total_items if total_items > 0 else 0

        return {
            'final_result': {
                'total_sum': total_sum,
                'total_items': total_items,
                'average_value': avg_value,
                'all_processed_values': all_processed_values,
                'num_mappers_used': len(mapped_results),
                'processing_time': processing_time
            },
            'processing_log': [
                f'Reducer: Combined {len(mapped_results)} chunks',
                f'Total items processed: {total_items}',
                f'Final sum: {total_sum}',
                f'Processing completed in {processing_time:.2f}s'
            ],
            'status': 'completed'
        }

    # Build the graph structure
    graph.add_node('distributor', distribute_data)
    graph.add_node('reducer', reduce_results)
    graph.set_entry_point('distributor')

    # Add mapper nodes dynamically (we'll add them during execution)
    # For now, add a reasonable number of potential mappers
    max_mappers = 10
    for i in range(max_mappers):
        mapper_func = create_mapper(i)
        graph.add_node(f'mapper_{i}', mapper_func)
        graph.add_edge(f'mapper_{i}', 'reducer')  # All mappers feed into reducer

    return graph.compile()

def run_map_reduce_example():
    """Run the map-reduce example with sample data"""

    print("🚀 GraphFlow Map-Reduce Example")
    print("=" * 50)

    # Create the map-reduce processor
    processor = create_map_reduce_processor()

    # Test with different datasets
    test_cases = [
        {
            'name': 'Small Dataset',
            'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'chunk_size': 3
        },
        {
            'name': 'Medium Dataset', 
            'data': list(range(1, 21)),  # 1 to 20
            'chunk_size': 4
        },
        {
            'name': 'Large Dataset',
            'data': list(range(1, 51)),  # 1 to 50
            'chunk_size': 7
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Input: {test_case['data'][:10]}{'...' if len(test_case['data']) > 10 else ''}")
        print(f"Total items: {len(test_case['data'])}, Chunk size: {test_case['chunk_size']}")

        # Run the map-reduce process
        start_time = time.time()
        result = processor.invoke({
            'input_data': test_case['data'],
            'chunk_size': test_case['chunk_size']
        })
        execution_time = time.time() - start_time

        # Display results
        final_result = result['final_result']
        print(f"\n✅ Results:")
        print(f"   Total Sum: {final_result['total_sum']}")
        print(f"   Items Processed: {final_result['total_items']}")
        print(f"   Average Value: {final_result['average_value']:.2f}")
        print(f"   Mappers Used: {final_result['num_mappers_used']}")
        print(f"   Parallel Processing Time: {final_result['processing_time']:.2f}s")
        print(f"   Total Execution Time: {execution_time:.2f}s")

        # Show processing log
        print(f"\n📝 Processing Log:")
        for log_entry in result['processing_log']:
            print(f"   {log_entry}")

def run_performance_comparison():
    """Compare parallel map-reduce vs sequential processing"""

    print("\n🏆 Performance Comparison: Parallel vs Sequential")
    print("=" * 60)

    # Test data
    test_data = list(range(1, 101))  # 1 to 100
    chunk_size = 10

    # Parallel processing with GraphFlow
    print("🔄 Running Parallel Map-Reduce...")
    processor = create_map_reduce_processor()

    parallel_start = time.time()
    parallel_result = processor.invoke({
        'input_data': test_data,
        'chunk_size': chunk_size
    })
    parallel_time = time.time() - parallel_start

    # Sequential processing for comparison
    print("⏳ Running Sequential Processing...")

    sequential_start = time.time()
    time.sleep(0.2 * (len(test_data) // chunk_size))  # Simulate sequential processing
    sequential_processed = [x ** 2 for x in test_data]
    sequential_sum = sum(sequential_processed)
    sequential_time = time.time() - sequential_start

    # Results comparison
    print(f"\n📊 Performance Results:")
    print(f"Dataset Size: {len(test_data)} items")
    print(f"Chunk Size: {chunk_size}")
    print(f"Number of Chunks: {len(test_data) // chunk_size}")
    print(f"\nParallel Processing:")
    print(f"  Time: {parallel_time:.2f}s")
    print(f"  Result Sum: {parallel_result['final_result']['total_sum']}")
    print(f"  Mappers Used: {parallel_result['final_result']['num_mappers_used']}")
    print(f"\nSequential Processing:")
    print(f"  Time: {sequential_time:.2f}s") 
    print(f"  Result Sum: {sequential_sum}")

    if sequential_time > parallel_time:
        speedup = sequential_time / parallel_time
        print(f"\n🚀 Parallel processing was {speedup:.1f}x faster!")
    else:
        print(f"\n⚡ Results completed in similar time (overhead for small datasets)")

if __name__ == "__main__":
    # Run the map-reduce examples
    run_map_reduce_example()

    # Run performance comparison
    run_performance_comparison()

    print("\n" + "=" * 60)
    print("🎯 Key Takeaways:")
    print("• Map-Reduce pattern enables parallel processing of large datasets")
    print("• GraphFlow automatically handles work distribution and result aggregation")
    print("• State reducers ensure safe parallel state updates")
    print("• Dynamic fan-out adapts to data size automatically")
    print("• Significant performance gains for CPU-intensive processing")
    print("=" * 60)