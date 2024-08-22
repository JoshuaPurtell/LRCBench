from apropos.src.bench.base import Benchmark, Question
from apropos.src.core.programs.dag import LM_DAG, DagRecord
import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, List


def match(record_a, record_b):
    return (
        record_a["counterparty_name"] == record_b["counterparty_name"]
        and record_a["amount"] == -record_b["amount"]
        and abs((record_a["date"] - record_b["date"]).days) <= 3
    )

def check_response(response: str, answer: Dict) -> bool:
    for key in answer:
        if key not in response or str(answer[key]) not in response:
            return False
    return True

def create_synthetic_datum(k=100,trial=0) -> Tuple[List[Dict], Dict]:
    random.seed(trial)
    counterparty_names = [
        "Walmart", "Amazon", "Apple", "CVS Health", "UnitedHealth Group",
        "ExxonMobil", "Berkshire Hathaway", "Alphabet", "McKesson", "AmerisourceBergen",
        "Chevron", "JPMorgan Chase", "Walgreens Boots Alliance", "Costco",
        "Cardinal Health", "Microsoft", "Kroger", "Ford Motor", "Citigroup",
        "AT&T", "General Motors", "Marathon Petroleum", "Anthem", "Fannie Mae",
        "Comcast", "Phillips 66", "Valero Energy", "Dell Technologies", "Target",
        "Bank of America", "Home Depot", "Boeing", "Wells Fargo", "Procter & Gamble",
        "FedEx", "Johnson & Johnson", "Humana", "Archer Daniels Midland", "Intel",
        "IBM", "State Farm Insurance", "Albertsons", "MetLife", "PepsiCo",
        "United Parcel Service", "Prudential Financial", "Walt Disney", "Sysco",
        "HP", "Cisco Systems", "Pfizer", "Lowe's", "Lockheed Martin", "FedEx",
        "Caterpillar", "Coca-Cola", "HCA Healthcare", "Energy Transfer",
        "Goldman Sachs Group", "Morgan Stanley", "Raytheon Technologies",
        "Abbott Laboratories", "AbbVie", "Delta Air Lines", "Charter Communications",
        "New York Life Insurance", "American Express", "Nationwide",
        "Best Buy", "Liberty Mutual Insurance Group", "Merck", "Tyson Foods",
        "TIAA", "Oracle", "General Dynamics", "TJX", "Nike", "World Fuel Services",
        "American Airlines Group", "Massachusetts Mutual Life Insurance",
        "ConocoPhillips", "Deere", "Tech Data", "Enterprise Products Partners",
        "Publix Super Markets", "General Electric", "Northrop Grumman",
        "Raytheon", "Plains GP Holdings", "3M", "AIG", "Progressive",
        "Arrow Electronics", "Centene", "United Continental Holdings",
        "PBF Energy", "Freddie Mac", "Hewlett Packard Enterprise", "Honeywell International"
    ]
    synthetic_data = []
    for _ in range(k):
        cnp = random.choice(counterparty_names)
        date = datetime(2023, random.randint(1, 12), random.randint(1, 28))
        date_delta = random.randint(0,3)
        amount = random.randint(1000, 10000)
        synthetic_data.append({"counterparty_name": cnp, "amount": amount, "date": (date).strftime("%Y-%m-%d")})
        synthetic_data.append({"counterparty_name": cnp, "amount": -amount, "date": (date + timedelta(days=date_delta)).strftime("%Y-%m-%d")})
    valid_needle = False
    needle = None
    while not valid_needle:
        needle = {"counterparty_name": random.choice(counterparty_names), "amount": random.randint(1000, 10000), "date": (datetime(2023, random.randint(1, 12), random.randint(1, 28))).strftime("%Y-%m-%d")}
        valid_needle = not any(match(needle, record) for record in synthetic_data)
    synthetic_data.append(needle)
    random.shuffle(synthetic_data)
    return synthetic_data, needle



class FinancialRecordsQuestion(Question):
    def __init__(self, haystack: List[Dict], needle: Dict):
        self.information = {
            "question": f"Financial records: "+"\n".join(f"{data['counterparty_name']} {data['amount']} {data['date']}" for data in haystack),
            "answer": needle,
            "haystack_size": len(haystack)//2
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
        correctness = check_response(str(answer), self.information["answer"])
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
        correctness = check_response(str(answer), self.information["answer"])
        return correctness, dag_record


class FinancialRecordsBenchmark(Benchmark):
    train: List[FinancialRecordsQuestion]
    test: List[FinancialRecordsQuestion]
    dev: List[FinancialRecordsQuestion]

    def __init__(self, haystack_sizes: List[int], k: int):
        self.train = []
        self.test = []
        self.dev = []
        for haystack_size in haystack_sizes:
            for trial in range(k):
                haystack, needle = create_synthetic_datum(haystack_size, trial)
                self.train.append(FinancialRecordsQuestion(haystack, needle))
                self.test.append(FinancialRecordsQuestion(haystack, needle))
                self.dev.append(FinancialRecordsQuestion(haystack, needle))
    
    def get_data_for_size(self, haystack_size: int, split: str) -> List[FinancialRecordsQuestion]:
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
    

