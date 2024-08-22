# LRCBench

Evals meant to evaluate language models' ability to reason over long contexts.

Currently, we support 3 settings with similar objectives:
- Coding: The model is given a coding question and a set of helper functions. It must select which (3) helper functions solve the problem.
- Transaction Matching: The model is given a set of accounting records, from which all but one can be paired according to the following criteria: a pair of records have opposite sign / same magnitude amounts, the same counterparty, and a date within 4 days. It must return the unpaired record.
- 2-cycle multiplication: The model is given a set of 2-cycles (from undergraduate group theory), and is asked to return the product of the two cycles in simplified form. (Note: Currently, LM models perform surprisingly poorly on this task.)

# Benchmarks

## Helper Function Invocation - Data Science
| LM | Score (Largest haystack size at which P(success) >= 0.5) |
|:----------|---------------:|
| Claude-3-opus-20240229 | 200+ |
| Claude-3-5-sonnet-20240620 | 140 |
| GPT-4(-32k) | 40 |
| Gemini-1.5-pro | 20 |
| Gemini-1.5-flash | 20 |
| GPT-4o-2024-08-06 | 20 |
| GPT-4o-mini | 10 |
| GPT-4-turbo | 5 |

## Transaction Matching
| LM | Score (Largest N pairs at which P(success) >= 0.5) |
|:----------|---------------:|
| Claude-3-5-sonnet-20240620 | 160 |
| Gemini-1.5-pro | 140 |
| Claude-3-opus-20240229 | 140 |
| GPT-4 | 40 |
| Gemini-1.5-flash | 30 |
| GPT-4o-turbo | 30 |
| GPT-4o-2024-08-06 | 20 |
| GPT-4o-mini | 10 |

## 2-cycle multiplication (Possibly implemented poorly?)
| LM | Score (Largest N 2-cycles at which P(success) >= 0.5) |
|:----------|---------------:|
| Claude-3-5-sonnet-20240620 | 2 |
| GPT-4o-turbo | 2 |
| GPT-4o-2024-08-06 | 2 |
| GPT-4o-mini | 2 |
| GPT-4 | 2 |
| Claude-3-opus-20240229 | 1 |
| Gemini-1.5-pro | 0 |
| Gemini-1.5-flash | 0 |
