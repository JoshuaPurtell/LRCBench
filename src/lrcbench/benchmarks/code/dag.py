from apropos.src.core.programs.convenience_functions.dag_constructors import (
    build_single_step_program,
)
from apropos.src.core.programs.prompt import (
    PromptTemplate,
    SystemMessage,
    Topic,
    UserMessage,
)


def code_problem_single_step(model_name="gpt-3.5-turbo"):
    execute = PromptTemplate(
        name="Code Function Identification",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="Premise",
                    topic_template="# Premise\nYou will be provided with a question about a tabular dataset and a collection of function definitions.\n$PROBLEM_DESCRIPTION",
                    instructions_fields={
                        "PROBLEM_DESCRIPTION": "The question requires applying three specific transformations to the dataset, plus one additional operation, to arrive at the answer. Each of the three transformations is critical for obtaining the correct solution.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="Objective",
                    topic_template="# Objective\nIdentify the three function IDs that correspond to the transformations needed to solve the given problem. Respond only with the list of function IDs.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            constraints=[
                Topic(
                    topic_name="Constraints",
                    topic_template="# Constraints\n- Each function is wrapped in XML tags with its ID as the tag name.\n- The question is provided in a <prompt> tag.\n- You must return exactly three function IDs.\n- Do not include any explanations or additional text in your response.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="User Input",
                    topic_template="<<<HAYSTACK>>>\nYour answer:",
                    instructions_fields={},
                    input_fields=["<<<HAYSTACK>>>"],
                )
            ]
        ),
        response_type="str",
        response_model_scheme=None,
        demonstrations=[],
    )
    code_problem_dag = build_single_step_program(
        execute,
        model_name=model_name,
        dag_input_names=["<<<HAYSTACK>>>"],
        dag_input_aliases={
            "question": "<<<HAYSTACK>>>",
        },
        dag_output_aliases={"<<<ANSWER>>>": "answer"},
    )
    return code_problem_dag
