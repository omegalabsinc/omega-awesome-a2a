from model import GPT2MediumModel
import time
import psutil
import os

def benchmark():
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024

    print("Starting model load...")
    start_time = time.time()
    model = GPT2MediumModel()
    load_time = time.time() - start_time

    load_mem = process.memory_info().rss / 1024 / 1024
    mem_usage = load_mem - start_mem

    prompts = [
        "The future of AI is",
        "Once upon a time",
        "The best way to learn is"
    ]

    print("\nRunning generation tests...")
    for prompt in prompts:
        start_time = time.time()
        output = model.generate(prompt)
        gen_time = time.time() - start_time
        print(f"\nPrompt: {prompt}")
        print(f"Generated [{gen_time:.2f}s]: {output}")

    print(f"\nBenchmark Results:")
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"Memory Usage: {mem_usage:.2f} MB")

if __name__ == "__main__":
    benchmark()
