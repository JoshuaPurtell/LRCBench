# LRCBench

Evals meant to evaluate language models' ability to reason over long contexts.

# Benchmarks

## Coding
TBD (will involve using a combination of helper functions)

## Transaction Matching
| LM | Score (Largest N pairs at which pass @ 10 >= 5) |
|:----------|---------------:|
| Claude-3-5-sonnet-20240620 | 160 |
| Claude-3-opus-20240229 | 140 |
| GPT-4 | 40 |
| GPT-4o-turbo | 30 |
| GPT-4o-2024-08-06 | 20 |
| GPT-4o-mini | 10 |

## 2-cycle multiplication (Possibly implemented poorly?)
| LM | Score (Largest N 2-cycles at which pass @ 10 >= 5) |
|:----------|---------------:|
| Claude-3-5-sonnet-20240620 | 2 |
| Claude-3-opus-20240229 | 1 |
| GPT-4 | 2 |
| GPT-4o-turbo | 2 |
| GPT-4o-2024-08-06 | 2 |
| GPT-4o-mini | 2 |
