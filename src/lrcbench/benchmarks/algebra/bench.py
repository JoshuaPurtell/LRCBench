from apropos.src.bench.base import Benchmark, Question, QABenchmark
from apropos.src.core.programs.dag import LM_DAG, DagRecord
import random
from typing import Tuple, Dict, List
from sympy.combinatorics import Permutation, PermutationGroup

def create_synthetic_datum(k=5, trial=0) -> Tuple[List[Permutation], Permutation]:
    random.seed(trial)
    S4 = PermutationGroup(Permutation(0, 1), Permutation(0, 1, 2, 3))
    all_elements = [Permutation(p) for p in S4.generate_schreier_sims(af=True)]
    def is_two_cycle(p):
        cycles = p.cyclic_form
        return len(cycles) == 1 and len(cycles[0]) == 2
    
    two_cycles = [p for p in all_elements if is_two_cycle(p)]
    
    haystack = random.sample(two_cycles, k=min(k, len(two_cycles)))
    needle = Permutation()
    for cycle in haystack:
        needle = needle * cycle
    return haystack, needle

def compare_permutations(answer_str: str, correct_permutation: Permutation) -> bool:
    cycle_str = answer_str.split("\n")[-1].strip()
    last_alpha_index = max((i for i, c in enumerate(cycle_str) if c.isalpha()), default=-1)
    if last_alpha_index != -1:
        cycle_str = cycle_str[last_alpha_index + 1:].strip()

    def parse_cycle_notation(cycle_str: str) -> Permutation:
        cycle_str = cycle_str.strip()
        if cycle_str == "()" or cycle_str == "":
            return Permutation()
        cycles = []
        for cycle in cycle_str.strip("()").split(")("):
            elements = [e.replace(")", "").replace("(", "").replace(".", "").replace(":","").strip() for e in cycle.replace(",", " ").split()]
            elements = [e for e in elements if e != ""]
            if len(elements) > 1:
                cycles.append(tuple(map(int, elements)))
        return Permutation(cycles)

    parsed_answer = parse_cycle_notation(cycle_str)
    parsed_correct = parse_cycle_notation(str(correct_permutation))
    equivalent = parsed_answer.array_form == parsed_correct.array_form
    return equivalent

class AlgebraQuestion(Question):
    def __init__(self, haystack: List[Permutation], needle: Permutation):
        self.information = {
            "question": " * ".join(str(p) for p in haystack),
            "answer": str(needle),
            "haystack_size": len(haystack)
        }
        self.correctness = False

    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert len(unique_inputs) == 1, f"There should be exactly one input edge, instead got {unique_inputs}"
        assert unique_inputs[0] == "<<<HAYSTACK>>>", f"The input edge should be for the haystack, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
 
        correctness = compare_permutations(answer, self.information["answer"])
        return correctness, dag_record

    async def compute_and_score_attempt(
        self,
        lm_dag: LM_DAG,
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
    ) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert len(unique_inputs) == 1, f"There should be exactly one input edge, instead got {unique_inputs}"
        assert unique_inputs[0] == "<<<HAYSTACK>>>", f"The input edge should be for the haystack, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert len(output_edges) == 1, f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = await lm_dag.arun(
            {"question": self.information["question"]}, verbose=True
        )
        system_message, user_message = list(lm_dag.nodes.values())[0].transform.prompt.compile(
            {"<<<HAYSTACK>>>": self.information["question"]}
        )
        answer = output["answer"]
        correctness = compare_permutations(answer, self.information["answer"])
        return correctness, dag_record

class AlgebraBenchmark(QABenchmark):
    train: List[AlgebraQuestion]
    test: List[AlgebraQuestion]
    dev: List[AlgebraQuestion]

    def __init__(self, haystack_sizes: List[int], k: int):
        self.train = []
        self.test = []
        self.dev = []
        for haystack_size in haystack_sizes:
            for trial in range(k):
                haystack, needle = create_synthetic_datum(haystack_size, trial)
                self.train.append(AlgebraQuestion(haystack, needle))
                self.test.append(AlgebraQuestion(haystack, needle))
                self.dev.append(AlgebraQuestion(haystack, needle))

    def get_data_for_size(self, haystack_size: int, split: str) -> List[AlgebraQuestion]:
        if split == "train":
            return [
                question for question in self.train if question.information["haystack_size"] == haystack_size
            ]
        elif split == "test":
            return [
                question for question in self.test if question.information["haystack_size"] == haystack_size
            ]
        elif split == "dev":
            return [
                question for question in self.dev if question.information["haystack_size"] == haystack_size
            ]
        else:
            raise ValueError(f"Invalid split: {split}")
        
if __name__ == "__main__":
    benchmark = AlgebraBenchmark(haystack_sizes=[2], k=10)
    questions = benchmark.get_data_for_size(2, "train")
    print(questions[0].information)