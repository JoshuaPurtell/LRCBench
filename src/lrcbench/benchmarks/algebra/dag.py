from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_single_step_program,
)
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    UserMessage,
    Topic,
)
from pydantic import BaseModel

# class PermutationResponse(BaseModel):
#     answer: str


def algebra_problem_single_step(model_name="gpt-3.5-turbo"):
    execute = PromptTemplate(
        name="Permutation Multiplication",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="Premise",
                    topic_template="# Premise\nYou will be given a list of 2-cycles from the symmetric group S4 under left multiplication. These are written in cycle notation, where each cycle represents a permutation of the elements {0, 1, 2, 3}.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="Objective",
                    topic_template="# Objective\nMultiply the given 2-cycles and provide the result as a single permutation in cycle notation. For example, if given (1 2) (2 3) (3 4), should observe that 4 -> 3 -> 2 -> 1, 3 -> 4,  3 -> 2, 1 -> 2, and so the result is (1 4 3 2).",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            constraints=[
                Topic(
                    topic_name="Constraints",
                    topic_template="# Constraints\n- Use cycle notation for the final answer. Leave only the final answer on the last line of your response. \n- Simplify the result to its shortest form",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="User Input",
                    topic_template="# Permutations\n<<<HAYSTACK>>>\nYour answer:",
                    instructions_fields={},
                    input_fields=["<<<HAYSTACK>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,  # PermutationResponse.schema()
        demonstrations=[],
    )
    algebra_problem_dag = build_single_step_program(
        execute,
        model_name=model_name,
        dag_input_names=["<<<HAYSTACK>>>"],
        dag_input_aliases={
            "question": "<<<HAYSTACK>>>",
        },
        dag_output_aliases={"<<<ANSWER>>>": "answer"},
    )
    return algebra_problem_dag
