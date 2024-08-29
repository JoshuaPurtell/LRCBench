# LRCBench

Evals meant to evaluate language models' ability to reason over long contexts.

Currently, we support 3 settings with similar objectives:
- Coding: The model is given a coding question and a set of helper functions. It must select which (3) helper functions solve the problem.
- Transaction Matching: The model is given a set of accounting records, from which all but one can be paired according to the following criteria: a pair of records have opposite sign / same magnitude amounts, the same counterparty, and a date within 4 days. It must return the unpaired record.
- 2-cycle multiplication: The model is given a set of 2-cycles (from undergraduate group theory), and is asked to return the product of the two cycles in simplified form. (Note: Currently, LM models perform surprisingly poorly on this task.)

# Benchmarks

In order to make these scores more robust to noise, we stop after 2 consecutive runs that fail to reach 0.6.

## Helper Function Invocation - Data Science
| LM | Best Score | Best Streak | Scores |
|:----------|---------------:|:----------|:----------|
| Gemini-1.5-pro | 180 | 0 | 767777667766765
| GPT-4(-32k) | 80 | 0 | 8775665755
| Claude-3-opus-20240229 | 70 | 0 | 878866645
| Gemini-1.5-flash | 60 | 0 | 98766745
| GPT-4-turbo | 50 | 0 | 6676645
| Claude-3-5-sonnet-20240620 | 30 | 0 | 96655
| GPT-4o-mini | 10 | 0 | 654
| GPT-4o-2024-08-06 | 0 | 0 | 5
| gemini-1.5-flash-8b-exp-0827 | 0 | 0 | 3

## Transaction Matching
| LM | Best Score | Best Streak | Scores |
|:----------|---------------:|---------------:|:----------|
| Gemini-1.5-pro | 200 | 20 | TTTT78899877783844
| Claude-3-5-sonnet-20240620 | 180 | 80 | TTTTTTTTTT8T78634
| Claude-3-opus-20240229 | 160 | 60 | 9TT89TTT98787723
| GPT-4(-32k) | 120 | 20 | TTTT8977976955
| Gemini-1.5-flash | 40 | 10 | TTT6623
| GPT-4-turbo | 40 | 10 | TTT7652
| GPT-4o-2024-08-06 | 30 | 10 | TTT823
| GPT-4o-mini | 20 | 5 | TT853
| gemini-1.5-flash-8b-exp-0827 | 10 | 5 | TT42

* Previously 40 (gpt-4 only), now 100 (gpt-4-32k)

## 2-cycle multiplication (Possibly implemented poorly?)
| LM | Best Score | Best Streak | Scores |
|:----------|---------------:|:----------|:----------|
| GPT-4-turbo | 3 | 1 | T800
| Claude-3-5-sonnet-20240620 | 2 | 1 | T42
| Claude-3-opus-20240229 | 2 | 1 | T41
| GPT-4o-2024-08-06 | 2 | 1 | T12
| GPT-4o-mini | 2 | 1 | T50
| GPT-4(-32k) | 2 | 1 | T31
| Gemini-1.5-pro | 0 | 0 | 0
| Gemini-1.5-flash | 0 | 0 | 0
| gemini-1.5-flash-8b-exp-0827 | 0 | 0 | 5

* Previously 2 (gpt-4 only), now 1 (gpt-4-32k)