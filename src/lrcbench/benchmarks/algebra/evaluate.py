from src.lrcbench.benchmarks.algebra.dag import algebra_problem_single_step
from src.lrcbench.benchmarks.algebra.bench import AlgebraBenchmark
from apropos.src.core.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG
from apropos import LM_DAG
import asyncio
from typing import List


async def score_for_size(
    haystack_size: int, dag: LM_DAG, benchmark: AlgebraBenchmark, k: int
):
    questions = benchmark.get_data_for_size(haystack_size, "train")
    correctnesses = await asyncio.gather(
        *[question.compute_and_score_attempt(dag) for question in questions[:k]]
    )
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses)


async def run_evaluation(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: AlgebraBenchmark, k: int
):
    largest_passing_haystack_size = 0
    success_rates = []
    print("Success rates: ", end="")
    for haystack_size in haystack_sizes:
        #try:
        success_rate = await score_for_size(haystack_size, dag, benchmark, k)
        print(f"Success rate for haystack size {haystack_size}: {success_rate}")
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


async def main(
    model_name: str, haystack_sizes: List[int], benchmark: AlgebraBenchmark, k: int
):
    base_dag = algebra_problem_single_step(model_name=model_name)
    demo_dag = algebra_problem_single_step(model_name="gpt-4o-mini")
    await run_evaluation(haystack_sizes, base_dag, benchmark, k=k)


if __name__ == "__main__":
    haystack_sizes = [1, 2, 3, 4, 5,6,7,8,9, 10, 15, 20, 25, 30]#, 50, 100
    k_for_success_rate = 10
    benchmark = AlgebraBenchmark(haystack_sizes=haystack_sizes, k=10)

    model_name = "o1-mini"

    # Maybe bootstrapping would help?
    asyncio.run(
        main(
            model_name=model_name,
            haystack_sizes=haystack_sizes,
            benchmark=benchmark,
            k=k_for_success_rate,
        )
    )

    # Largest passing haystack size (no demos):
    # o1-preview: | 768887766
    # o1-mini: | TT9TT897977666...
    # Claude-3-5-sonnet-20240620: 3 | 1 | T632
    # Claude-3-opus-20240229: 3 | 1 | T810
    # GPT-4: 2 | 1 | T54
    # GPT-4o-turbo: 3 | 1 | T752
    # GPT-4o-2024-08-06: 3 | 1 | T632
    # GPT-4o-mini: 2 | 1 | T34
    # Gemini-1.5-pro: 0 | 0 | 0
    # Gemini-1.5-flash: 0 | 0 | 0
