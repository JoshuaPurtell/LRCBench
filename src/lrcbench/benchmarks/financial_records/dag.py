from apropos.src.core.programs.convenience_functions.dag_constructors import build_single_step_program
from apropos.src.core.programs.prompt import PromptTemplate, SystemMessage, UserMessage, Topic
from pydantic import BaseModel
# # Premise
# You will be provided with records of accounting entries. Some represent real-world transactions, and others represent offsetting entries.
# Each real-world transaction ought to have an offsetting entry to balance the books.
# Matching entries share the following characteristics:
# - Same counterparty name
# - Same date
# - Amount with the same absolute value but opposite sign
# ## Examples of Matching Entries
# ### Matching Pair 1
# Google 1000 2023-01-01
# Google -1000 2023-01-01
# ### Matching Pair 2
# Apple 2000 2023-02-01
# Apple -2000 2023-02-01
# ### Matching Pair 3
# Microsoft 3000 2023-03-01
# Microsoft -3000 2023-03-01
# # Objective
# Identify the entry that does not have an offsetting entry. Respond only with its information, in the same format as it is presented.
# """, "The entries you have to pick from:" + stringified_haystack

class RecordResponse(BaseModel):
    counterparty_name: str
    amount: int
    date: str

def records_problem_single_step(model_name="gpt-3.5-turbo"):
    execute = PromptTemplate(
        name="Record Matching",
        system=SystemMessage(
            premise=[
                Topic(
                    topic_name="Premise",
                    topic_template="# Premise\nYou will be provided with records of accounting entries.\n $RECORDS_DESCRIPTION\n\n $PROBLEM_DESCRIPTION",
                    instructions_fields={
                        "RECORDS_DESCRIPTION": "Some represent real-world transactions, and others represent offsetting entries. Each real-world transaction ought to have an offsetting entry to balance the books. Matching entries share the following characteristics:\n- Same counterparty name\n- Date within 4 days\n- Amount with the same absolute value but opposite sign",
                        "PROBLEM_DESCRIPTION": "One record does not have an offsetting entry.",
                    },
                    input_fields=[],
                )
            ],
            objective=[
                Topic(
                    topic_name="Objective",
                    topic_template="# Objective\nIdentify the entry that does not have an offsetting entry. Respond only with its information, in the same format as it is presented.",
                    instructions_fields={},
                    input_fields=[],
                )
            ],
            constraints=[
            ],
        ),
        user=UserMessage(
            user=[
                Topic(
                    topic_name="User Input",
                    topic_template="# Records\n<<<HAYSTACK>>>\nYour answer:",
                    instructions_fields={},
                    input_fields=["<<<HAYSTACK>>>"],
                )
            ]
        ),
        response_type="pydantic",
        response_model_scheme=RecordResponse.schema(),
        demonstrations=[],
    )
    math_problem_dag = build_single_step_program(
            execute,
            model_name=model_name,
            dag_input_names=["<<<HAYSTACK>>>"],
            dag_input_aliases={
                "question": "<<<HAYSTACK>>>",
            },
            dag_output_aliases={"<<<ANSWER>>>": "answer"},
        )
    return math_problem_dag