## Adapters
| File                                 | Model   | Platform    | Config | Steps | LoRA Rank |
|---                                   |---      |---          |---     |---    |---        |
| ```qwen-sft-gt```                    | qwen    | M5          | Local  | -     | -         |
| ```qwen-sft-ps```                    | qwen    | M5          | Local  | -     | -         |
| ```llama-sft-gt_20260419_023444```   | llama   | M5          | Local  | 1500  | 8         |
| ```llama-sft-ps_20260419_085139```   | llama   | M5          | Local  | 1500  | 8         |
| ```llama-sft-gt_20260419_091436```   | llama   | Runpod L40S | Cloud  | 5000  | 64        |
| ```llama-sft-ps_20260420_033833```   | llama   | Runpod L40S | Cloud  | 5000  | 64        |

## Inference Results
| File                                    | Model | Platform | Adapters Used |
|---                                      |---    |---       |---            |
| ```infer_results_20260413_230752.csv``` | qwen  | M5       | qwen-sft-*    |
| ```infer_results_20260418_175012.csv``` | qwen  | M5       | qwen-sft-*    |
