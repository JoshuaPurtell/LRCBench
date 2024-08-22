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
    correctnesses = await asyncio.gather(
        *[
            question.compute_and_score_attempt(dag)
            for question in random.sample(questions, k)
        ]
    )
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses)


async def main(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: FinancialRecordsBenchmark, k: int
):
    largest_passing_haystack_size = 0
    for haystack_size in haystack_sizes:
        try:
            pass_at_k = await score_for_size(haystack_size, dag, benchmark, k)
            print(f"P[Success] for haystack size {haystack_size}: {pass_at_k}")
            if pass_at_k <= 0.5:
                break
            largest_passing_haystack_size = haystack_size
        except Exception as e:
            print(f"Error for haystack size {haystack_size}: {e}")
            break
    print(
        f"Largest passing (P[Success]) haystack size: {largest_passing_haystack_size}"
    )
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
    k_for_pass_at_k = 10
    benchmark = FinancialRecordsBenchmark(
        haystack_sizes=haystack_sizes, k=k_for_pass_at_k
    )
    dag = records_problem_single_step(model_name="gemini-1.5-flash")
    asyncio.run(main(haystack_sizes, dag, benchmark, k_for_pass_at_k))

    # Largest passing haystack size:
    # Claude-3-5-sonnet-20240620: 160
    # Claude-3-opus-20240229: 140
    # GPT-4: 40 # Error - swaps amount and name - {'counterparty_name': '3969', 'amount': 'Nationwide', 'date': '2023-02-22'}
    # GPT-4o-turbo: 30
    # GPT-4o-2024-08-06: 20
    # GPT-4o-mini: 10
