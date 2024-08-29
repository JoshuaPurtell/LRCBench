import asyncio
import random
from typing import List
import os
import json
from apropos import LM_DAG
from src.lrcbench.benchmarks.code.bench import CodeBenchmark, generate_prompts
from src.lrcbench.benchmarks.code.dag import code_problem_single_step


import tiktoken
gpt_4_tokenizer = tiktoken.encoding_for_model("gpt-4")

async def score_for_size(
    haystack_size: int, dag: LM_DAG, benchmark: CodeBenchmark
):
    questions = benchmark.get_data_for_size(haystack_size, "train")
    random.seed(0)
    correctnesses = await asyncio.gather(
        *[
            question.compute_and_score_attempt(dag)
            for question in questions
        ]
    )
    token_counts = [len(gpt_4_tokenizer.encode(question.information["question"])) for question in questions]
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses), int(sum(token_counts) // len(token_counts))


async def main(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: CodeBenchmark
):
    largest_passing_haystack_size = 0
    success_rates = []
    mean_token_counts = []
    print("Success rates: ", end="")
    for haystack_size in haystack_sizes:
        success_rate, mean_token_count = await score_for_size(haystack_size, dag, benchmark)
        success_rates.append(success_rate)
        mean_token_counts.append(mean_token_count)
        rate = int(success_rate * 10)
        if success_rate == 1.0:
            print(f"\033[94mT\033[0m", end="")
        elif success_rate >= 0.9:
            print(f"\033[94m{rate}\033[0m", end="")
        elif success_rate >= 0.7:
            print(f"\033[92m{rate}\033[0m", end="")
        elif success_rate >= 0.5:
            print(f"\033[93m{rate}\033[0m", end="")
        else:
            print(f"\033[91m{rate}\033[0m", end="")
        if max(success_rates[-2:]) < 0.6:
            for i in range(2):
                if len(success_rates) > 0:
                    success_rates.pop()
            break
        largest_passing_haystack_size = haystack_size
    print(
        f"\nLargest passing haystack size: {largest_passing_haystack_size}"
    )
    print(f"Mean token counts: {mean_token_counts}")

    perfect_score_size = 0
    for size, rate in zip(haystack_sizes, success_rates):
        if rate == 1.0:
            perfect_score_size = size
        else:
            break
    print(f"Largest haystack size with streak: {perfect_score_size}")
    return largest_passing_haystack_size


async def run_evaluation(model_name: str):
    haystack_sizes = [
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        140,
        160,
        180,
        200
    ]
    if not os.path.exists("temp/prompts.json"):
        prompts = await generate_prompts(num_questions=50)
        with open("temp/prompts.json", "w") as f:
            json.dump(prompts, f)
    else:
        with open("temp/prompts.json", "r") as f:
            prompts = json.load(f)
    print(f"Running evaluation for {model_name}")
    benchmark = CodeBenchmark(prompts, haystack_sizes=haystack_sizes)
    dag = code_problem_single_step(model_name=model_name)
    await main(haystack_sizes, dag, benchmark)

async def run_for_models(models: List[str]):
    for model in models:
        await run_evaluation(model_name=model)

if __name__ == "__main__":

    models = ["gpt-4-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4-32k","claude-3-5-sonnet-20240620","claude-3-opus-20240229","gemini-1.5-pro", "gemini-1.5-flash"]#,"gemini-1.5-flash-8b-exp-0827"
    asyncio.run(run_for_models(models))

    # Make it hit the cache

    # claude-3-opus-20240229: 50 | 0 | 8933 | 7776645
    # claude-3-5-sonnet-20240620: 30 | 0 | 6393 | 87655
    # gpt-4-32k: 30 | 0 | 6393 | 76655
    # gemini-1.5-pro: 20 | 0 | 4900 | 6655
    # gemini-1.5-flash: 20 | 0 | 4900 | 7654
    # gpt-4-turbo: 0 | 0 | 0 | 5
    # gpt-4o-2024-08-06: 10 | 0 | 1123 | 655
    # gpt-4o-mini: 10 | 0 | 1123 | 744
