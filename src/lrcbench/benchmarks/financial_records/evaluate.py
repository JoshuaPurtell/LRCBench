import asyncio
import random
from typing import List

from apropos import LM_DAG
from src.lrcbench.benchmarks.financial_records.bench import FinancialRecordsBenchmark
from src.lrcbench.benchmarks.financial_records.dag import records_problem_single_step


async def score_for_size(
    haystack_size: int, dag: LM_DAG, benchmark: FinancialRecordsBenchmark, k: int
):
    random.seed(0)
    questions = benchmark.get_data_for_size(haystack_size, "train")
    correctnesses = []
    for question in random.sample(questions, k):
        try:
            result = await question.compute_and_score_attempt(dag)
            correctnesses.append(result)
        except Exception:
            correctnesses.append((0, None))
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses)


async def main(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: FinancialRecordsBenchmark, k: int
):
    largest_passing_haystack_size = 0
    success_rates = []
    print("Success rates: ", end="")
    for haystack_size in haystack_sizes:
        try:
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
            largest_passing_haystack_size = haystack_size
        except Exception as e:
            print(f"\nError for haystack size {haystack_size}: {e}")
            break
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


if __name__ == "__main__":
    haystack_sizes = [
        2,
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
        200,
        250,
        300,
        400,
        500,
        750,
        1000,
        1500,
    ]
    k_for_success_rate = 10
    benchmark = FinancialRecordsBenchmark(
        haystack_sizes=haystack_sizes, k=k_for_success_rate
    )
    dag = records_problem_single_step(model_name="gemini-1.5-flash-8b-exp-0827")
    asyncio.run(main(haystack_sizes, dag, benchmark, k_for_success_rate))

    # New Results
    # Sonnet: 180 | 80 | TTTTTTTTTT8T78634
    # Opus: 160 | 60 |  9TT89TTT98787723
    # GPT-4: 120 | 20 | TTTT8977976955
    # Gemini-1.5-pro: 200 | 20 | TTTT78899877783844
    # Gemini-1.5-flash: 40 | 10 | TTT6623
    # GPT-4o-turbo: 40 | 10 | TTT7652
    # GPT-4o-2024-08-06: 30 | 10 | TTT823
    # GPT-4o-mini: 20 | 5 | TT853
    # Gemini-1.5-flash-8b-exp-0827: 10 | 5 | TT42

    # Largest passing haystack size:
    # Claude-3-5-sonnet-20240620: 160
    # Gemini-1.5-pro: 140
    # Claude-3-opus-20240229: 140
    # GPT-4: 40
    # Gemini-1.5-flash: 30
    # GPT-4o-turbo: 30
    # GPT-4o-2024-08-06: 20
    # llama3-70b-8192: 10
    # GPT-4o-mini: 10
    # gemini-1.5-pro-exp-0827: 2
    # gemini-1.5-flash-exp-0827: 2
    # Gemini-1.5-flash-8b-exp-0827: 2 => 5
