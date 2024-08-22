import asyncio
import random
from typing import List

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
    for haystack_size in haystack_sizes:
        try:
            pass_at_k = await score_for_size(haystack_size, dag, benchmark, k)
            print(f"P[Success] for haystack size {haystack_size}: {pass_at_k}")
            if pass_at_k < 0.5:
                break
            largest_passing_haystack_size = haystack_size
        except Exception as e:
            print(f"Error for haystack size {haystack_size}: {e}")
            break
    print(
        f"Largest passing (P[Success]) haystack size: {largest_passing_haystack_size}"
    )
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
    prompts = await generate_prompts(num_questions=50)
    benchmark = CodeBenchmark(prompts, haystack_sizes=haystack_sizes)
    dag = code_problem_single_step(model_name="gpt-4-turbo")
    await main(haystack_sizes, dag, benchmark, k_for_pass_at_k)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

    # Make it hit the cache

    # claude-3-opus-20240229: 200+ (will create more data later)
    # claude-3-5-sonnet-20240620: 140
    # gpt-4(-32k): 40
    # gpt-4-turbo: 5
    # gpt-4o-2024-08-06: 20
    # gpt-4o-mini: 10
