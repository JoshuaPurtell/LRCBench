import asyncio
import random
from typing import List
import os
import json
from apropos import LM_DAG
from src.lrcbench.benchmarks.code.bench import CodeBenchmark, generate_prompts
from src.lrcbench.benchmarks.code.dag import code_problem_single_step


async def score_for_size(
    haystack_size: int, dag: LM_DAG, benchmark: CodeBenchmark, k: int
):
    questions = benchmark.get_data_for_size(haystack_size, "train")
    random.seed(0)
    correctnesses = await asyncio.gather(
        *[
            question.compute_and_score_attempt(dag)
            for question in random.sample(questions, k)
        ]
    )
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses)


async def main(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: CodeBenchmark, k: int
):
    largest_passing_haystack_size = 0
    success_rates = []
    print("Success rates: ", end="")
    for haystack_size in haystack_sizes:
        # try:
        success_rate = await score_for_size(haystack_size, dag, benchmark, k)
        success_rates.append(success_rate)
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
        # if success_rate <= 0.5:
        #     break
        largest_passing_haystack_size = haystack_size
        # except Exception as e:
        #     print(f"\nError for haystack size {haystack_size}: {e}")
        #     break
    print(
        f"\nLargest passing (> .5 P[Success]) haystack size: {largest_passing_haystack_size}"
    )

    perfect_score_size = 0
    for size, rate in zip(haystack_sizes, success_rates):
        if rate == 1.0:
            perfect_score_size = size
        else:
            break
    print(f"Largest haystack size with perfect score: {perfect_score_size}")
    return largest_passing_haystack_size


async def run_evaluation():
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
    ]  # 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
    k_for_pass_at_k = 10
    # prompts = await generate_prompts(num_questions=50)
    if not os.path.exists("temp/prompts.json"):
        prompts = await generate_prompts(num_questions=20)
        with open("temp/prompts.json", "w") as f:
            json.dump(prompts, f)
    else:
        with open("temp/prompts.json", "r") as f:
            prompts = json.load(f)
    benchmark = CodeBenchmark(prompts, haystack_sizes=haystack_sizes)
    dag = code_problem_single_step(model_name="gemini-1.5-flash-8b-exp-0827")
    await main(haystack_sizes, dag, benchmark, k_for_pass_at_k)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

    # Make it hit the cache

    # claude-3-opus-20240229: 70 | 0 | 878866645
    # claude-3-5-sonnet-20240620: 30 | 0 | 96655
    # gpt-4-32k: 80 | 0 | 8775665755
    # gemini-1.5-pro: 180 | 0 | 767777667766765
    # gemini-1.5-flash: 60 | 0 | 98766745
    # gpt-4-turbo: 50 | 0 | 6676645
    # gpt-4o-2024-08-06: 0 | 0 | 5
    # gpt-4o-mini: 10 | 0 | 654
    # gemini-1.5-flash-8b-exp-0827: 0 | 0 | 3
