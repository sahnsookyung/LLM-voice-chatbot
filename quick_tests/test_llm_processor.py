import sys
import os
import time
import statistics

# Add the parent directory to the path to import llm_processor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llm_processor import LLMProcessor
except ImportError as e:
    print(f"❌ Error importing LLMProcessor: {e}")
    print("Please ensure this script is in a subdirectory and llm_processor.py is in the parent directory.")
    sys.exit(1)


def run_single_test(llm_processor, prompt: str):
    """
    Runs a single streaming and one-time generation test.
    Returns the timing data.
    """
    # Run One-Time Generation
    start_time_one_time = time.time()
    response_one_time = llm_processor.generate_one_time_response(prompt)
    end_time_one_time = time.time()
    one_time_duration = end_time_one_time - start_time_one_time

    # Run Streaming Generation
    start_time_streaming = time.time()
    first_token_time = None
    response_generator = llm_processor.generate_response(prompt)

    for i, chunk in enumerate(response_generator):
        if i == 0:
            first_token_time = time.time() - start_time_streaming
        # Consume the generator, no need to build the full response here
        pass

    end_time_streaming = time.time()
    streaming_total_duration = end_time_streaming - start_time_streaming

    return {
        'one_time': one_time_duration,
        'streaming_total': streaming_total_duration,
        'first_token': first_token_time
    }


def run_multiple_tests_for_model(model_name: str, prompt: str, num_runs: int):
    """
    Runs both streaming and one-time generation tests multiple times for a single model
    and prints the average results.
    """
    print("==================================================")
    print(f"  Testing Model: {model_name}")
    print("==================================================\n")

    # Construct the correct path to the personality file
    script_dir = os.path.dirname(__file__)
    # Convert the relative path to an absolute path for robustness
    personality_file_path = os.path.abspath(os.path.join(script_dir, '..', 'personality-template.txt'))

    try:
        llm_processor = LLMProcessor(model_name=model_name, personality_file=personality_file_path)
    except Exception as e:
        print(f"❌ Could not initialize LLMProcessor for model '{model_name}': {e}")
        return

    one_time_durations = []
    streaming_total_durations = []
    first_token_times = []

    print(f"Running {num_runs} tests for {model_name}. This may take some time...")
    for i in range(num_runs):
        print(f"Test {i + 1}/{num_runs}...", end='\r', flush=True)
        results = run_single_test(llm_processor, prompt)
        one_time_durations.append(results['one_time'])
        streaming_total_durations.append(results['streaming_total'])
        if results['first_token'] is not None:
            first_token_times.append(results['first_token'])

    avg_one_time = statistics.mean(one_time_durations)
    avg_streaming_total = statistics.mean(streaming_total_durations)

    if first_token_times:
        avg_first_token = statistics.mean(first_token_times)
    else:
        avg_first_token = None

    print(f"\n\n--- Timing Comparison Summary for {model_name} ({num_runs} runs) ---")

    if avg_first_token is not None:
        print(f"Average Time to First Token (Streaming): {avg_first_token:.4f} seconds")
    else:
        print("Average Time to First Token (Streaming): N/A")

    print(f"Average Total Duration (Streaming):      {avg_streaming_total:.4f} seconds")
    print(f"Average Total Duration (One-Time):       {avg_one_time:.4f} seconds\n")


if __name__ == "__main__":
    test_prompt = "Chicken or egg?"
    num_tests = 1

    models_to_test = [
        "hf.co/mlabonne/gemma-3-12b-it-abliterated-v2-GGUF:Q4_K_M",
        "hf.co/mlabonne/gemma-3-4b-it-abliterated-v2-GGUF:Q4_K_M",
        "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q4_K_M"
    ]

    for model in models_to_test:
        run_multiple_tests_for_model(model, test_prompt, num_tests)
