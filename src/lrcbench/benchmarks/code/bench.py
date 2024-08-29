import asyncio
import inspect
import random
import sys
import uuid
from typing import Dict, List, Tuple
import os
import json
from apropos import LLM
from apropos.src.bench.base import Benchmark, Question
from apropos.src.core.programs.dag import LM_DAG, DagRecord
from src.lrcbench.benchmarks.code.helpers import *
from src.lrcbench.benchmarks.code.helpers import BASE_DATAFRAME
import random

random.seed(42)


import hashlib


def generate_deterministic_id(name):
    return hashlib.md5(name.encode()).hexdigest()


FUNCTION_INFO = [
    {
        "name": name,
        "id": generate_deterministic_id(name),
        "definition": inspect.getsource(obj),
    }
    for i, (name, obj) in enumerate(
        inspect.getmembers(sys.modules["src.lrcbench.benchmarks.code.helpers"])
    )
    if inspect.isfunction(obj)
    and obj.__module__ == "src.lrcbench.benchmarks.code.helpers"
]


async def create_synthetic_datum(
    k: int = 3, instance: int = 0
) -> Tuple[List[Dict], str, str]:
    random.seed(instance)
    selected_transforms = random.sample(FUNCTION_INFO, k)

    transform_names = [transform["name"] for transform in selected_transforms]
    transform_ids = [transform["id"] for transform in selected_transforms]
    transform_definitions = [
        transform["definition"] for transform in selected_transforms
    ]

    aliased_definitions = []
    for name, id, definition in zip(
        transform_names, transform_ids, transform_definitions
    ):
        aliased_definition = definition.replace(name, f"f_{id}")
        aliased_definitions.append(aliased_definition)

    system_message = """You will be given the head of a tabular dataset and the definition of 3 functions.
Please write a question that requires applying these three transformations to the dataset, plus one additional operation, to arrive at the answer.
Ensure that each of the three transformations is critical for obtaining the correct solution.

This question will be used to evaluate the performance of a model on a tabular dataset.
Therefore, do NOT giveaway the transformations necessary to obtain the answer in your question.

Moreover, ensure that your question cannot be easily answered simply by looking at the function names or at variable names in the function definitions. It should require the model to reason about the transformations in the context of the dataset.

For instance, this prompt is bad because it can be answered simply by looking at the column name defined in the function (m_and_a_attractiveness)
```
Given the dataset, determine which startup has the highest potential for long-term success in its industry. To do this, you need to consider the startup's potential to set industry standards, its attractiveness for mergers and acquisitions, and its potential for strategic partnerships. Additionally, filter the startups to only include those that have a positive growth rate. Which startup meets these criteria and has the highest overall potential for long-term success?
</prompt>
<haystack>
<53983c69-5d9a-43f4-ad63-f8312ad205e4>
def f_53983c69-5d9a-43f4-ad63-f8312ad205e4(df):
    strategic_value = df['innovation_index']
    financial_health = df['runway_months'] / 12
    market_position = df['estimated_market_share'].rank(pct=True)
    synergy_potential = df['ecosystem_impact_score']
    return df.assign(m_and_a_attractiveness=(strategic_value + financial_health + market_position + synergy_potential) / 4)
</53983c69-5d9a-43f4-ad63-f8312ad205e4>
```
"""
    user_message = f"""Here is the head of the dataset:
{BASE_DATAFRAME.head().to_markdown()}

Here are the definitions of the functions:
1. {aliased_definitions[0]}
2. {aliased_definitions[1]}
3. {aliased_definitions[2]}

Here are the names of the transformations:
{transform_names}

Your Question:
"""
    prompt = await LLM("claude-3-5-sonnet-20240620").async_respond(
        system_prompt=system_message,
        user_prompt=user_message,
    )
    return prompt, transform_ids


def build_haystack(prompt, transform_ids, FUNCTION_INFO, haystack_size: int, seed=0):
    transforms = [
        func_info for func_info in FUNCTION_INFO if func_info["id"] in transform_ids
    ]
    non_transforms = [
        func_info for func_info in FUNCTION_INFO if func_info["id"] not in transform_ids
    ]
    random.seed(seed)
    hay = random.sample(non_transforms, k=haystack_size)
    haystack_transforms = transforms + hay
    hay_stringified = ""
    for straw in haystack_transforms:
        aliased_definition = straw["definition"].replace(
            straw["name"], f"f_{straw['id']}"
        )
        hay_stringified += f"<{straw['id']}>\n{aliased_definition}\n</{straw['id']}>\n"
    haystack = f"""
<prompt>
{prompt}
</prompt>
<haystack>
{hay_stringified}
</haystack>
"""
    return haystack, transform_ids


def evaluate_code_answer(answer: str, correct_answer: List[str]) -> bool:
    return all(correct_answer_id in answer for correct_answer_id in correct_answer)


class CodeQuestion(Question):
    def __init__(self, transforms: List[Dict], question: str, haystack_size: int):
        self.information = {
            "answer": transforms,
            "question": question,
            "haystack_size": haystack_size,
        }
        self.correctness = False

    def compute_and_score_attempt_sync(self, lm_dag: LM_DAG) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert (
            len(unique_inputs) == 1
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert (
            unique_inputs[0] == "<<<HAYSTACK>>>"
        ), f"The input edge should be for the haystack, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = lm_dag.run_standard(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        correctness = evaluate_code_answer(str(answer), self.information["answer"])
        return correctness, dag_record

    async def compute_and_score_attempt(
        self,
        lm_dag: LM_DAG,
        frozen_params: Dict = {"dag_record": None, "unfrozen_node_names": []},
    ) -> Tuple[bool, DagRecord]:
        unique_inputs = list(
            set([edge[0][1] for edge in lm_dag.edges if edge[0][0] == "DAG_INPUT"])
        )
        assert (
            len(unique_inputs) == 1
        ), f"There should be exactly one input edge, instead got {unique_inputs}"
        assert (
            unique_inputs[0] == "<<<HAYSTACK>>>"
        ), f"The input edge should be for the haystack, instead got {unique_inputs[0]}"
        output_edges = [edge for edge in lm_dag.edges if edge[1][0] == "DAG_OUTPUT"]
        assert (
            len(output_edges) == 1
        ), f"There should be exactly one output edge, instead got {len(output_edges)}"
        output, dag_record = await lm_dag.arun(
            {"question": self.information["question"]}, verbose=True
        )
        answer = output["answer"]
        correctness = evaluate_code_answer(str(answer), self.information["answer"])
        return correctness, dag_record


class CodeBenchmark(Benchmark):
    train: List[CodeQuestion]
    test: List[CodeQuestion]
    dev: List[CodeQuestion]

    def __init__(self, prompts: List[Dict], haystack_sizes: List[int]):
        self.train = []
        self.test = []
        self.dev = []

        all_questions = []
        for prompt in prompts:
            for haystack_size in haystack_sizes:
                question, _ = build_haystack(
                    prompt["question"],
                    prompt["answer"],
                    FUNCTION_INFO,
                    haystack_size=haystack_size,
                )
                all_questions.append(
                    CodeQuestion(
                        transforms=prompt["answer"],
                        question=question,
                        haystack_size=haystack_size,
                    )
                )
        self.train = all_questions[: int(0.6 * len(all_questions))]
        self.dev = all_questions[
            int(0.6 * len(all_questions)) : int(0.8 * len(all_questions))
        ]
        self.test = all_questions[int(0.8 * len(all_questions)) :]

    def get_split(self, split: str) -> List[CodeQuestion]:
        if split == "train":
            return self.train
        elif split == "dev":
            return self.dev
        elif split == "test":
            return self.test
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_data_for_size(self, haystack_size: int, split: str) -> List[CodeQuestion]:
        questions = self.get_split(split)
        return [
            question
            for question in questions
            if question.information["haystack_size"] == haystack_size
        ]


async def generate_prompts(num_questions: int, seed: int = 42) -> List[Dict]:
    random.seed(seed)

    async def create_prompt(i, k=3):
        prompt, ids = await create_synthetic_datum(k=k, instance=i)
        return {"question": prompt, "answer": ids}

    prompts = []
    for i in range(num_questions):
        prompts.append(await create_prompt(i))
    return prompts


if __name__ == "__main__":
    if not os.path.exists("temp/prompts.json"):
        prompts = asyncio.run(generate_prompts(num_questions=50))
        with open("temp/prompts.json", "w") as f:
            json.dump(prompts, f)
    else:
        with open("temp/prompts.json", "r") as f:
            prompts = json.load(f)
    benchmark = CodeBenchmark(prompts, haystack_sizes=[10, 20, 30, 50])
