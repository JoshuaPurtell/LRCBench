from src.lrcbench.benchmarks.algebra.dag import algebra_problem_single_step
from src.lrcbench.benchmarks.algebra.bench import AlgebraBenchmark
from apropos.src.core.optimizers.baselines.bffsrs import BreadthFirstRandomSearch_DAG
from apropos import LM_DAG
import asyncio
from typing import List

async def score_for_size(haystack_size: int, dag: LM_DAG, benchmark: AlgebraBenchmark, k: int):
    questions = benchmark.get_data_for_size(haystack_size, "train")
    correctnesses = await asyncio.gather(*[question.compute_and_score_attempt(dag) for question in questions[:k]])
    return sum([correctness for correctness, _ in correctnesses]) / len(correctnesses)

async def run_evaluation(
        haystack_sizes: List[int],
        dag: LM_DAG,
        benchmark: AlgebraBenchmark,
        k: int
):
    largest_passing_haystack_size = 0
    for haystack_size in haystack_sizes:
        pass_at_k = await score_for_size(haystack_size, dag, benchmark, k)
        print(f"Pass@K for haystack size {haystack_size}: {pass_at_k}")
        if pass_at_k < 0.5:
            break
        largest_passing_haystack_size = haystack_size
    print(f"Largest passing (> .5 pass@k) haystack size: {largest_passing_haystack_size}")
    return largest_passing_haystack_size

async def main(
        model_name: str,

        haystack_sizes: List[int],
        benchmark: AlgebraBenchmark,
        k: int
):
    base_dag = algebra_problem_single_step(
        model_name=model_name
    )
    demo_dag = algebra_problem_single_step(
        model_name="gpt-4o-mini"
    )
    await run_evaluation(haystack_sizes, base_dag, benchmark, k=k)
    optimizer = BreadthFirstRandomSearch_DAG(
        student_program=demo_dag,
        teacher_program=demo_dag,
        dataset_handler=benchmark,
        cfg = {"optimization": {
                "combination_size": 5,
                "n_iterations": 5,
                "program_search_parallelization_factor": 2,
                "validation_size": 10,
                "test_size": 10,
            },
            "bootstrapping": {
                "patches": ["C"],
                "n_questions": 50,
            },
            "verbose": True,
        }
    )
    bootstrapped_dag_3_2cycles = await optimizer.optimize_demonstrations(
    )
    optimizer = BreadthFirstRandomSearch_DAG(
        student_program=demo_dag,
        teacher_program=base_dag,
        dataset_handler=benchmark,
        cfg = {"optimization": {
                "combination_size": 3,
                "n_iterations": 5,
                "program_search_parallelization_factor": 2,
                "validation_size": 10,
                "test_size": 10,
            },
            "bootstrapping": {
                "patches": ["D"],
                "n_questions": 99,
            },
            "verbose": True,
        }
    )
    bootstrapped_dag_4_2cycles = await optimizer.optimize_demonstrations(
    )
    for node_name in base_dag.nodes:
        four_cycles_demos = bootstrapped_dag_4_2cycles.nodes[node_name].transform.prompt.demonstrations
        three_cycles_demos = bootstrapped_dag_3_2cycles.nodes[node_name].transform.prompt.demonstrations
        base_dag.nodes[node_name].transform.prompt.demonstrations = four_cycles_demos[0:3] + three_cycles_demos[0:3]
    
    await run_evaluation(haystack_sizes, base_dag, benchmark, k=k)

if __name__ == "__main__":
    haystack_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    k_for_pass_at_k = 10
    benchmark = AlgebraBenchmark(haystack_sizes=haystack_sizes, k=100)
    
    model_name = "gpt-4"
    
    # Maybe bootstrapping would help?
    asyncio.run(main(
        model_name=model_name,
        haystack_sizes=haystack_sizes,
        benchmark=benchmark,
        k=k_for_pass_at_k
    ))
    
    # Largest passing haystack size (no demos): 
    # Claude-3-5-sonnet-20240620: 2
    # Claude-3-opus-20240229: 1
    # GPT-4: 2
    # GPT-4o-turbo: 2
    # GPT-4o-2024-08-06: 2
    # GPT-4o-mini: 2

    # Largest passing haystack size (with 5 demos):
    # Claude-3-5-sonnet-20240620: 2
    # Claude-3-opus-20240229: 2
    # GPT-4: 1
    # GPT-4o-turbo: 3
    # GPT-4o-2024-08-06: 3
    # GPT-4o-mini: 3

    
    

