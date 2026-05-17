## Adapters
| File                                 | Model | Platform       | Config     | Steps | LoRA Rank | Final Loss | Training Time (hrs) | 
|---                                   |---    |---             |---         |---    |---        | ---        | ---                 |
| ```qwen-sft-gt```                    | qwen  | M5             | Local      | 100   | -         | -          | -                   |
| ```qwen-sft-ps```                    | qwen  | M5             | Local      | 100   | -         | -          | -                   |
| ```llama-sft-gt_20260419_023444```   | llama | M5             | Local      | 1500  | 8         | 1.845546   | 6.16                |
| ```llama-sft-ps_20260419_085139```   | llama | M5             | Local      | 1500  | 8         | 0.340333!  | 6.30                |
| ```llama-sft-gt_20260419_091436```   | llama | Runpod L40S    | Cloud      | 5000  | 64        | 1.855181   | 18.4                |
| ```llama-sft-ps_20260420_033833```   | llama | Runpod L40S    | Cloud      | 5000  | 64        | 0.572233!  | 16.8                |
| ```llama-sft-wiki_20260512_082331``` | llama | Rupod H100 SXM | Cloud Wiki | 15625 | 64        | 0.032000!  | 10.6                |


note: 
- hyperparams are under ```./cfgs/```. 
- File names for the log csvs correspond to the adapters timestamp.
- PS and wiki SFT final losses indicate overfitting


## Inference Results
| File                                    | Model | Platform | Conditions                                       | Adapters Used                                | Notes                          |
|---                                      |---    |---       |---                                               |---                                           |---                             |
| ```infer_results_20260413_230752.csv``` | qwen  | M5       | base, qwen-sft-gt, qwen-sft-ps                   | qwen-sft-*                                   | Has bias?(binary) annotation   |
| ```infer_results_20260418_175012.csv``` | qwen  | M5       | base, qwen-sft-gt, qwen-sft-ps                   | qwen-sft-*                                   | Re-run, no annotation          |
| ```infer_results_20260426_004811.csv``` | llama | M5       | base, llama-sft-gt, llama-sft-ps                 | llama-sft-gt_20260419_091436 (cloud rank-64) | Cloud adapters, 60 runs / cond |
| ```infer_results_20260512_140118.csv``` | llama | M5       | base, llama-sft-gt, llama-sft-ps, llama-sft-wiki | llama-sft-wiki_20260512_082331               | -                              |

## Bias Scores
| File                              | Corpus                    | n   | Mean  | Rubric   |
|---                                |---                        |---  |---    |---       |
| bias_scores_gt.csv                | NELA-GT                   | 500 | 1.818 | 0 – 5    |
| bias_scores_ps.csv                | NELA-PS                   | 500 | 0.210 | 0 – 5    |
| bias_scores_wiki.csv              | Wikipedia HS              | 500 | 0.06  | 0 – 5    |
| infer_results_20260512_140118.csv | Completions (of 4 conds.) | 240 | —     | 0 – 5    |

## WEAT Results
Test 1 is HS Selectivity, Test 2 is Policy Necessity
| File                       | Condition      | Test | d      | p     | sig? |
|---                         |---             |---   |---     |---    |---   |
| `weat_20260513_013829.csv` | B              | 1    | +1.150 | 0.012 | *    |
|  ...                       | B              | 2    | −1.314 | 0.997 |      |
|  ...                       | GT             | 1    | +0.866 | 0.045 | *    |
|  ...                       | GT             | 2    | −1.408 | 0.997 |      |
|  ...                       | PS             | 1    | +0.825 | 0.052 |      |
|  ...                       | PS             | 2    | −1.402 | 0.997 |      |
|  ...                       | N              | 1    | +1.435 | 0.001 | **   |
|  ...                       | N              | 2    | −0.973 | 0.970 |      |

note:
- d is the effect size: more positive = elite school terms are more associated with positive attributes, which was what the hypotheses predicted
- significant for: * p < 0.05, * p < 0.01
- results under `weat/results/`
