import asyncio
import random
from typing import List

from apropos import LM_DAG
from src.lrcbench.benchmarks.financial_records.bench import FinancialRecordsBenchmark
from src.lrcbench.benchmarks.financial_records.dag import records_problem_single_step

import tiktoken
gpt_4_tokenizer = tiktoken.encoding_for_model("gpt-4")
random.seed(42)

async def score_for_size(
    haystack_size: int, dag: LM_DAG, benchmark: FinancialRecordsBenchmark, k: int
):
    questions = benchmark.get_data_for_size(haystack_size, "train")
    
    correctnesses = await asyncio.gather(
        *[
            question.compute_and_score_attempt(dag)
            for question in random.sample(questions, k)
        ]
    )
    token_counts = [len(gpt_4_tokenizer.encode(question.information["question"])) for question in questions]
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses), int(sum(token_counts) // len(token_counts))


async def main(
    haystack_sizes: List[int], dag: LM_DAG, benchmark: FinancialRecordsBenchmark, k: int
):
    largest_passing_haystack_size = 0
    success_rates = []
    mean_token_counts = []
    print("Success rates: ", end="")
    for haystack_size in haystack_sizes:
        try:
            success_rate, mean_token_count = await score_for_size(haystack_size, dag, benchmark, k)
            mean_token_counts.append(mean_token_count)
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
    print(f"Mean token count: {mean_token_counts[-1]}")

    perfect_score_size = 0
    for size, rate in zip(haystack_sizes, success_rates):
        if rate == 1.0:
            perfect_score_size = size
        else:
            break
    print(f"Largest haystack size with perfect score: {perfect_score_size}")
    return largest_passing_haystack_size

async def run_evaluation(model_name: str):
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
    dag = records_problem_single_step(model_name=model_name)
    await main(haystack_sizes, dag, benchmark, k_for_success_rate)

async def run_for_models(models: List[str]):
    for model in models:
        print(f"Running evaluation for {model}")
        await run_evaluation(model_name=model)


if __name__ == "__main__":
    models = ["gpt-4-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4-32k","claude-3-5-sonnet-20240620","claude-3-opus-20240229","gemini-1.5-pro", "gemini-1.5-flash"]#,"gemini-1.5-flash-8b-exp-0827"
    asyncio.run(run_for_models(models))

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
